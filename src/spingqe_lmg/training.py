"""GQE training loop with checkpoint/resume and CSV logging.

Loop structure adapted from ``train.py`` in Mindbeam-AI/SpinGQE (MIT):
each epoch the model generates a batch of gate sequences, their per-step
energies are measured, and the model trains on (sequence, energies) pairs
with the cumulative-logit loss.

Additions over the reference: config-driven setup, atomic checkpointing with
full RNG state (so SLURM requeues resume cleanly), energy caching/parallelism,
optional energy normalization for the sigmoid weighting (LMG energies scale
with N; the reference beta was tuned for a fixed 4-qubit scale), and
append-only CSV logs. On resume, a few epoch rows may repeat in the CSVs;
aggregation keeps the last occurrence.
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from spingqe_lmg import __version__
from spingqe_lmg.config import PennyLaneConfig, RunConfig, config_to_dict
from spingqe_lmg.enums import DeviceKind, EnergyNorm, InitState
from spingqe_lmg.evaluators import EvaluatorProtocol, build_evaluator
from spingqe_lmg.hamiltonians import build_hamiltonian, lmg_mean_field_angle
from spingqe_lmg.model import GQEModel, build_model
from spingqe_lmg.pools import build_pool


def initial_angle_for(cfg: RunConfig) -> float | str:
    """Reference-state spec implied by the config."""
    if cfg.train.init_state is InitState.ZERO:
        return 0.0
    if cfg.train.init_state is InitState.MEAN_FIELD:
        return lmg_mean_field_angle(cfg.hamiltonian.h, cfg.hamiltonian.lam)
    if cfg.train.init_state is InitState.GHZ_X:
        return "ghz_x"
    raise ValueError(f"unknown init_state: {cfg.train.init_state!r}")


@dataclass
class TrainResult:
    best_energy: float
    best_epoch: int
    ground_energy: float
    best_tokens: list[int]
    out_dir: Path
    refined_energy: float | None = None


@dataclass
class _TrainContext:
    pool: list
    evaluator: EvaluatorProtocol
    model: GQEModel
    opt: torch.optim.Optimizer
    ground_energy: float
    e_init: float
    beta: float
    n_qubits: int
    device: str
    init_angle: float | str


def _build_context(cfg: RunConfig) -> _TrainContext:
    tc = cfg.train
    n_qubits = cfg.hamiltonian.n_qubits

    set_seed(tc.seed)
    device = resolve_device(tc.device)

    pool = build_pool(cfg.pool, n_qubits)
    init_angle = initial_angle_for(cfg)
    ham = build_hamiltonian(cfg.hamiltonian)
    evaluator, ground_energy = build_evaluator(cfg, pool, ham, init_angle, n_qubits)
    e_init = evaluator.initial_energy()
    beta = tc.beta

    model = build_model(cfg.model, len(pool) + 1, tc.seq_len).to(device)
    opt = model.configure_optimizers(
        weight_decay=tc.weight_decay,
        learning_rate=tc.learning_rate,
        betas=(tc.adam_beta1, tc.adam_beta2),
        device_type="cuda" if device == "cuda" else "cpu",
    )

    return _TrainContext(
        pool=pool,
        evaluator=evaluator,
        model=model,
        opt=opt,
        ground_energy=ground_energy,
        e_init=e_init,
        beta=beta,
        n_qubits=n_qubits,
        device=device,
        init_angle=init_angle,
    )


def _restore_checkpoint(
    ckpt_path: Path, model: GQEModel, opt: torch.optim.Optimizer, device: str
) -> tuple[int, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    opt.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt["epoch_next"]
    best = ckpt["best"]
    torch.set_rng_state(ckpt["rng"]["torch"].cpu())
    if device == "cuda" and ckpt["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["rng"]["cuda"]])
    np.random.set_state(ckpt["rng"]["numpy"])
    random.setstate(ckpt["rng"]["random"])
    print(f"resumed from {ckpt_path} at epoch {start_epoch}")
    return start_epoch, best


def _measure(
    tokens: torch.Tensor,
    ctx: _TrainContext,
    energy_norm: EnergyNorm,
) -> tuple[np.ndarray, np.ndarray]:
    gen_inds = (tokens[:, 1:] - 1).cpu().numpy()
    if (gen_inds < 0).any():
        raise ValueError("BOS token at non-initial position in generated sequence")
    raw = ctx.evaluator(gen_inds)
    return raw, _normalize(raw, energy_norm, ctx.e_init, ctx.n_qubits)


def _run_epochs(
    ctx: _TrainContext,
    start_epoch: int,
    best: dict,
    out_dir: Path,
    tc,
) -> tuple[dict, int]:
    ckpt_dir = out_dir / "checkpoints"
    losses_csv = out_dir / "losses.csv"
    eval_csv = out_dir / "eval.csv"

    for epoch in range(start_epoch, tc.epochs + 1):
        ctx.model.eval()
        with torch.no_grad():
            tokens, _ = ctx.model.generate(
                n_sequences=tc.seq_gen,
                max_new_tokens=tc.seq_len,
                temperature=tc.temperature,
                device=ctx.device,
            )
        raw, normed = _measure(tokens, ctx, tc.energy_norm)
        energies = torch.from_numpy(normed).float().to(ctx.device)

        final_raw = raw[:, -1]
        i_min = int(np.argmin(final_raw))
        if final_raw[i_min] < best["energy"]:
            best = {
                "energy": float(final_raw[i_min]),
                "epoch": epoch,
                "tokens": [int(t) for t in tokens[i_min].cpu()],
            }

        ctx.model.train()
        perm = torch.randperm(len(tokens), device=tokens.device)
        token_batches = torch.tensor_split(tokens[perm], tc.n_batches)
        energy_batches = torch.tensor_split(energies[perm], tc.n_batches)
        loss_values = []
        uw_loss_values = []
        batch_sizes = []
        for token_batch, energy_batch in zip(token_batches, energy_batches, strict=True):
            ctx.opt.zero_grad()
            loss, uw_loss = ctx.model.calculate_loss(token_batch, energy_batch, ctx.beta)
            loss.backward()
            ctx.opt.step()
            loss_values.append(loss.item())
            uw_loss_values.append(uw_loss.item())
            batch_sizes.append(len(token_batch))
        avg_loss = float(np.average(loss_values, weights=batch_sizes))
        avg_uw_loss = float(np.average(uw_loss_values, weights=batch_sizes))

        _append_csv(
            losses_csv,
            "epoch,loss,uw_loss,min_E,mean_E",
            f"{epoch},{avg_loss:.6g},{avg_uw_loss:.6g},"
            f"{final_raw.min():.10g},{final_raw.mean():.10g}",
        )
        print(
            f"epoch {epoch}: loss {avg_loss:.4f} (uw {avg_uw_loss:.4f}) "
            f"min_E {final_raw.min():.6f} best {best['energy']:.6f} exact {ctx.ground_energy:.6f}"
        )

        if epoch != 0 and epoch % tc.eval_iter == 0:
            ctx.model.eval()
            with torch.no_grad():
                eval_tokens, pred = ctx.model.generate(
                    n_sequences=tc.eval_sequences,
                    max_new_tokens=tc.seq_len,
                    temperature=tc.temperature,
                    device=ctx.device,
                )
            eval_raw, eval_normed = _measure(eval_tokens, ctx, tc.energy_norm)
            pred = pred.cpu().numpy().reshape(-1)
            true_final_raw = eval_raw[:, -1]
            mae = float(np.mean(np.abs(pred - eval_normed[:, -1])))

            gen_inds = (eval_tokens[:, 1:] - 1).cpu().numpy()
            if (gen_inds < 0).any():
                raise ValueError("BOS token at non-initial position in eval sequence")
            stutters = int(np.sum(gen_inds[:, 1:] == gen_inds[:, :-1]))
            stutter = (
                stutters / (gen_inds.shape[0] * (gen_inds.shape[1] - 1))
                if gen_inds.shape[1] > 1
                else 0.0
            )

            _append_csv(
                eval_csv,
                "epoch,mae,mean_E,min_E,stutter",
                f"{epoch},{mae:.6g},{true_final_raw.mean():.10g},"
                f"{true_final_raw.min():.10g},{stutter:.4f}",
            )
            print(
                f"  eval @ {epoch}: MAE {mae:.4f} mean_E {true_final_raw.mean():.6f} "
                f"min_E {true_final_raw.min():.6f} stutter {stutter:.3f}"
            )

        if tc.save_final_checkpoint and epoch % tc.checkpoint_every == 0:
            _atomic_save(
                {
                    "epoch_next": epoch + 1,
                    "model_state_dict": ctx.model.state_dict(),
                    "optimizer_state_dict": ctx.opt.state_dict(),
                    "best": best,
                    "rng": {
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if ctx.device == "cuda" else None,
                        "numpy": np.random.get_state(),
                        "random": random.getstate(),
                    },
                },
                ckpt_dir / "ckpt-latest.pt",
            )

    if tc.save_final_checkpoint:
        _atomic_save(
            {
                "epoch_next": tc.epochs + 1,
                "model_state_dict": ctx.model.state_dict(),
                "optimizer_state_dict": ctx.opt.state_dict(),
                "best": best,
                "rng": {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if ctx.device == "cuda" else None,
                    "numpy": np.random.get_state(),
                    "random": random.getstate(),
                },
            },
            ckpt_dir / "ckpt-latest.pt",
        )
    return best, tc.epochs


def _finalize(
    ctx: _TrainContext,
    cfg: RunConfig,
    best: dict,
    out_dir: Path,
    extra_metadata: dict | None = None,
) -> TrainResult:
    tc = cfg.train
    if tc.save_final_checkpoint:
        _atomic_save(
            {
                "model_state_dict": ctx.model.state_dict(),
                "optimizer_state_dict": ctx.opt.state_dict(),
            },
            out_dir / "final.pt",
        )

    _atomic_write_json(
        {**config_to_dict(cfg), "vocab_size": len(ctx.pool) + 1, "pool_size": len(ctx.pool)},
        out_dir / "config.json",
    )
    refined = None
    if tc.refine and best["tokens"]:
        pl = tc.pennylane or PennyLaneConfig()
        refined = ctx.evaluator.refine_tokens(
            best["tokens"],
            qml_device=pl.device,
            shots=pl.refine_shots if pl.refine_shots > 0 else 0,
            seed=tc.seed,
            optimizer=pl.refine_optimizer,
            gradient=pl.refine_gradient,
        )
        print(
            f"refined best sequence: {best['energy']:.6f} -> {refined.energy:.6f} "
            f"(exact {ctx.ground_energy:.6f}, {refined.n_evaluations} evaluations)"
        )

    best_tokens = best["tokens"][1:]
    if any(t == 0 for t in best_tokens):
        raise ValueError("BOS token at non-initial position in best sequence")
    best_ops = [ctx.pool[t - 1].label for t in best_tokens]
    best_record = {**best, "op_labels": best_ops, "ground_energy": ctx.ground_energy}
    if refined is not None:
        best_record["refined_energy"] = refined.energy
        best_record["refined_angles"] = refined.angles
    _atomic_write_json(best_record, out_dir / "best_sequence.json")
    metadata = {
        "run_name": cfg.run_name,
        "ground_energy": ctx.ground_energy,
        "initial_energy": ctx.e_init,
        "init_state": tc.init_state.value,
        "init_angle": ctx.init_angle,
        "shots": tc.pennylane.shots if tc.pennylane is not None else 0,
        "refine_shots": tc.pennylane.refine_shots if tc.pennylane is not None else 0,
        "best_energy": best["energy"],
        "best_epoch": best["epoch"],
        "abs_error": best["energy"] - ctx.ground_energy,
        "refined_energy": refined.energy if refined is not None else None,
        "refined_abs_error": (refined.energy - ctx.ground_energy) if refined is not None else None,
        "package_version": __version__,
        "torch_version": torch.__version__,
        "device": ctx.device,
        **(extra_metadata or {}),
    }
    _atomic_write_json(metadata, out_dir / "metadata.json")

    return TrainResult(
        best_energy=best["energy"],
        best_epoch=best["epoch"],
        ground_energy=ctx.ground_energy,
        best_tokens=best["tokens"],
        out_dir=out_dir,
        refined_energy=refined.energy if refined is not None else None,
    )


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(spec: DeviceKind) -> str:
    if spec is DeviceKind.AUTO:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return spec.value


def _append_csv(path: Path, header: str, row: str) -> None:
    new = not path.exists()
    with open(path, "a") as f:
        if new:
            f.write(header + "\n")
        f.write(row + "\n")


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_write_json(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _normalize(energies: np.ndarray, mode: EnergyNorm, e_init: float, n_qubits: int) -> np.ndarray:
    if mode is EnergyNorm.NONE:
        return energies
    if mode is EnergyNorm.SHIFT_SCALE:
        return (energies - e_init) / n_qubits
    raise ValueError(f"unknown energy_norm: {mode!r}")


def train(
    cfg: RunConfig, out_dir: str | Path, resume: bool = False, extra_metadata: dict | None = None
) -> TrainResult:
    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    plot_dir = out_dir / "plots"
    for d in (ckpt_dir, plot_dir):
        d.mkdir(parents=True, exist_ok=True)
    ctx = _build_context(cfg)

    ckpt_path = ckpt_dir / "ckpt-latest.pt"
    start_epoch, best = (
        _restore_checkpoint(ckpt_path, ctx.model, ctx.opt, ctx.device)
        if resume and ckpt_path.exists()
        else (0, {"energy": float("inf"), "epoch": -1, "tokens": []})
    )

    best, _epoch = _run_epochs(ctx, start_epoch, best, out_dir, cfg.train)
    return _finalize(ctx, cfg, best, out_dir, extra_metadata)
