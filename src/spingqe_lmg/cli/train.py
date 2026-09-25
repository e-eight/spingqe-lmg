"""Console entry points: spingqe-train, spingqe-refine."""

import argparse
import csv
import json
from pathlib import Path

import torch

from spingqe_lmg.config import (
    PennyLaneConfig,
    apply_overrides,
    config_from_dict,
    load_config,
    manifest_row_to_overrides,
)
from spingqe_lmg.evaluators import build_evaluator
from spingqe_lmg.hamiltonians import build_hamiltonian
from spingqe_lmg.model import build_model
from spingqe_lmg.pools import build_pool
from spingqe_lmg.training import initial_angle_for, train


def _check_backend_guard(cfg, allow_slow_backend: bool, assume_refine: bool = False) -> None:
    """Refuse the pure-Python ``default.qubit`` PennyLane device unless overridden.

    Refinement always runs through PennyLane regardless of the training
    evaluator, so a config that never sets ``train.pennylane.device`` silently
    falls back to ``default.qubit``. Both DeltaAI partitions are GPU-only, so a
    run stuck on this CPU simulator burns a GPU allocation for nothing.

    ``assume_refine`` is for ``refine_main``: that entry point always performs
    PennyLane refinement, independent of the training-time ``train.refine``
    flag, so it must not be gated on that flag the way ``train_main`` is.
    """
    if (not assume_refine and not cfg.train.refine) or allow_slow_backend:
        return
    pl = cfg.train.pennylane or PennyLaneConfig()
    if pl.device == "default.qubit":
        raise SystemExit(
            "refusing to run with the slow default.qubit PennyLane backend on DeltaAI: "
            "both partitions are GPU-only and default.qubit is a pure-Python CPU "
            "simulator, so this burns the GPU allocation for no benefit. Set "
            "train.pennylane.device to 'lightning.qubit' (or 'lightning.gpu'), or use "
            "the incremental evaluator (train.evaluator='incremental'/'incremental_gpu') "
            "for training. Pass --allow-slow-backend to override."
        )


def train_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train a GQE model from a TOML config.")
    parser.add_argument("--config", help="path to a run config TOML")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override a config value (repeatable)",
    )
    parser.add_argument("--out-dir", help="run output directory")
    parser.add_argument("--resume", action="store_true", help="resume from latest checkpoint")
    parser.add_argument("--manifest", help="sweep manifest CSV (use with --row)")
    parser.add_argument("--row", type=int, help="0-based row of the manifest to run")
    parser.add_argument(
        "--allow-slow-backend",
        action="store_true",
        help="allow the pure-Python default.qubit PennyLane device (otherwise refused)",
    )
    args = parser.parse_args(argv)

    if args.manifest is not None:
        if args.row is None:
            parser.error("--manifest requires --row")
        with open(args.manifest, newline="") as f:
            rows = list(csv.DictReader(f))
        if args.row >= len(rows):
            parser.error(f"--row {args.row} out of range (manifest has {len(rows)} rows)")
        row = rows[args.row]
        config_path = row["config"]
        overrides = manifest_row_to_overrides(row) + args.overrides
        out_dir = args.out_dir or row["out_dir"]
    else:
        if not (args.config and args.out_dir):
            parser.error("--config and --out-dir are required without --manifest")
        config_path = args.config
        overrides = args.overrides
        out_dir = args.out_dir

    cfg = load_config(config_path, overrides)
    _check_backend_guard(cfg, args.allow_slow_backend)
    extra = {}
    result = train(cfg, out_dir, resume=args.resume, extra_metadata=extra)
    print(
        f"done: best E = {result.best_energy:.8f} (epoch {result.best_epoch}), "
        f"exact E0 = {result.ground_energy:.8f}, error = "
        f"{result.best_energy - result.ground_energy:.3e}"
    )


def refine_main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Angle-refine the best sequence of a completed run (updates its JSON files)."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--generate",
        type=int,
        default=0,
        metavar="N",
        help="also sample N fresh sequences from final.pt and refine the best candidates",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="how many of the freshly generated candidates to refine (with --generate)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override a config value from the run's config.json (e.g. train.n_jobs=4)",
    )
    parser.add_argument(
        "--allow-slow-backend",
        action="store_true",
        help="allow the pure-Python default.qubit PennyLane device (otherwise refused)",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    cfg_data = json.loads((run_dir / "config.json").read_text())
    for extra in ("vocab_size", "pool_size"):
        cfg_data.pop(extra, None)
    if args.overrides:
        apply_overrides(cfg_data, args.overrides)
    cfg = config_from_dict(cfg_data)
    _check_backend_guard(cfg, args.allow_slow_backend, assume_refine=True)
    best = json.loads((run_dir / "best_sequence.json").read_text())

    n_qubits = cfg.hamiltonian.n_qubits
    pool = build_pool(cfg.pool, n_qubits)
    init_angle = initial_angle_for(cfg)

    ham = build_hamiltonian(cfg.hamiltonian)
    evaluator, ground_energy = build_evaluator(cfg, pool, ham, init_angle, n_qubits)

    def _refine(tokens):
        pl = cfg.train.pennylane or PennyLaneConfig()
        return evaluator.refine_tokens(
            tokens,
            qml_device=pl.device,
            shots=pl.shots if pl.refine_shots > 0 else 0,
            seed=cfg.train.seed,
            optimizer=pl.refine_optimizer,
            gradient=pl.refine_gradient,
        )

    result = _refine(best["tokens"])
    best_tokens = best["tokens"]

    if args.generate > 0:
        candidates = _generate_candidates(cfg, pool, run_dir, args.generate, args.top)
        for tokens in candidates:
            cand = _refine(tokens)
            print(f"  candidate: raw {cand.initial_energy:.6f} -> refined {cand.energy:.6f}")
            if cand.energy < result.energy:
                result = cand
                best_tokens = [int(t) for t in tokens]

    best["refined_energy"] = result.energy
    best["refined_angles"] = result.angles
    best["refined_tokens"] = best_tokens
    best["refined_op_labels"] = [pool[t - 1].label for t in best_tokens[1:]]
    (run_dir / "best_sequence.json").write_text(json.dumps(best, indent=2))
    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["refined_energy"] = result.energy
        meta["refined_abs_error"] = abs(result.energy - meta["ground_energy"])
        meta_path.write_text(json.dumps(meta, indent=2))

    grd = best.get("ground_energy")
    print(
        f"raw best E = {result.initial_energy:.8f} -> refined E = {result.energy:.8f} "
        f"(exact E0 = {grd:.8f}, error = {result.energy - grd:.3e}, "
        f"{result.n_evaluations} circuit evaluations)"
    )


def _generate_candidates(cfg, pool, run_dir: Path, n_generate: int, top: int):
    """Sample sequences from a trained model and return the ``top`` lowest-energy ones.

    Generates at the training temperature from final.pt, measures true final
    energies, keeps the best.
    """
    n_qubits = cfg.hamiltonian.n_qubits
    init_angle = initial_angle_for(cfg)
    model = build_model(cfg.model, len(pool) + 1, cfg.train.seq_len)
    state = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    with torch.no_grad():
        tokens, cumsum_logits = model.generate(
            n_sequences=n_generate,
            max_new_tokens=cfg.train.seq_len,
            temperature=cfg.train.temperature,
            device="cpu",
        )
    if top < n_generate:
        model_order = cumsum_logits.reshape(-1).argsort()
        tokens = tokens[model_order[: max(top * 2, top)]]
        n_measured = len(tokens)
    else:
        n_measured = n_generate
    ham = build_hamiltonian(cfg.hamiltonian)
    evaluator, _ = build_evaluator(cfg, pool, ham, init_angle, n_qubits)
    energies = evaluator((tokens[:, 1:] - 1).numpy())[:, -1]
    order = energies.argsort()[:top]
    print(
        f"generated {n_generate}, measured {n_measured}, best raw energies: {energies[order].round(4)}"
    )
    return [[int(t) for t in tokens[i]] for i in order]
