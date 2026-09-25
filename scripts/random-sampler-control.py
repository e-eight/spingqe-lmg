"""Compute-matched random-sampler control for the Table 1 pool-accuracy cells.

Answers the question the pool-accuracy table cannot answer on its own: does the
learned sequence model earn its keep, or would uniformly drawn token sequences
reach the same refined energy at the same evaluation budget?

Protocol, matched cell-by-cell to a paired GQE run:

  1. Mirror the paired run's ``config.json`` verbatim (pool, Hamiltonian,
     reference state, sequence length, refinement settings).
  2. Draw ``--n-samples`` uniform random length-L token sequences, score each
     one raw through the evaluator, and keep the lowest-energy sequence.
  3. Refine that single best sequence with the paired run's own refinement
     settings, exactly as ``training._finalize`` does.

Budget matching.  GQE's best sequence is selected from the training candidates
only (``training._run_epochs`` updates ``best`` from ``raw[:, -1]`` of each
epoch's ``seq_gen`` candidates).  With ``epochs=700`` the loop runs epochs
0..700 inclusive, so the pool is (700+1) * seq_gen = 7010 sequences.  The
100-sequence checkpoint evaluations every 50 epochs (1400 more) are diagnostic
and never update ``best``, so crediting the control with them would overpay it.
The primary budget is therefore 7010; ``--extra-samples`` additionally records
the best at 8410 so a reviewer can read the comparison either way from one run.

Scoring evaluator.  Raw scoring is evaluator-independent to float64 noise (the
2026-07-25 verification report measured dicke vs parity_even agreeing to
4.4e-14 on fixed sequences), and this control has no training loop for
per-step rounding differences to compound through.  Scoring therefore uses the
fastest admissible evaluator (mapping: collective -> dicke, pauli_pair ->
incremental) rather than whatever the legacy paired run used, and
``--verify-scoring`` cross-checks the two on real sequences so the substitution
is audited rather than assumed.  Refinement is *not* substituted: it mirrors
the paired run's device, optimizer, and gradient.

Usage (one cell):
    python scripts/random-sampler-control.py \
        --gqe-run <runs-root>/<sweep>/<run> \
        --draw 1 \
        --out-dir <runs-root>/<control-sweep>/<cell>
"""

import argparse
import dataclasses
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np

# The mirrored PennyLane evaluator used only by --verify-scoring emits one
# snapshot warning per sequence; the substituted evaluator does the real work.
warnings.filterwarnings("ignore", message="Snapshots are not supported")

from spingqe_lmg.config import PennyLaneConfig, config_from_dict, config_to_dict
from spingqe_lmg.enums import EvaluatorKind, PoolKind
from spingqe_lmg.evaluators import build_evaluator
from spingqe_lmg.hamiltonians import build_hamiltonian
from spingqe_lmg.pools import build_pool
from spingqe_lmg.training import initial_angle_for

# Candidates per epoch (TrainConfig.seq_gen) times epochs+1; see module docstring.
PRIMARY_BUDGET = 7010
EXTRA_BUDGET = 1400  # checkpoint-eval sequences, diagnostic-only in GQE

# D3 fastest-admissible scoring evaluator by pool kind.
_AUTO_SCORER = {
    PoolKind.COLLECTIVE: EvaluatorKind.DICKE,
    PoolKind.PAULI_PAIR: EvaluatorKind.INCREMENTAL,
}


def _draw_seed(pool_kind: str, n_qubits: int, lam: float, draw: int) -> int:
    """Deterministic, cell-independent RNG seed (no wall-clock, so reproducible)."""
    key = f"{pool_kind}|{n_qubits}|{lam!r}|{draw}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63)


# Pre-nesting TrainConfig keys, still present in the legacy sweeps that source
# Table 1's pairwise cells (lmg-t1-scaling-n12-16, pairwise-l12-uniform,
# lmg-extmf-gap-n12-16).  They map onto today's [train.pennylane] block.
# scripts/mine-resource-table.py handles the same split ad hoc; promoting this
# into config.py is a sync-back candidate.
_LEGACY_PENNYLANE_KEYS = {
    "qml_device": "device",
    "shots": "shots",
    "refine_shots": "refine_shots",
    "refine_optimizer": "refine_optimizer",
    "refine_gradient": "refine_gradient",
}


def _migrate_legacy_train(train: dict) -> dict | None:
    """Fold pre-nesting refinement keys into a [train.pennylane] dict.

    Returns the migration record (for metadata) or None if nothing was legacy.
    """
    found = {k: train.pop(k) for k in list(_LEGACY_PENNYLANE_KEYS) if k in train}
    if not found:
        return None
    nested = dict(train.get("pennylane") or {})
    for old, new in _LEGACY_PENNYLANE_KEYS.items():
        if old not in found:
            continue
        value = found[old]
        # Old schema wrote refine_shots as a bool; the field is an int shot count.
        if new in ("shots", "refine_shots") and isinstance(value, bool):
            value = int(value)
        nested.setdefault(new, value)
    train["pennylane"] = nested
    return {"legacy_keys": found, "migrated_to_pennylane": nested}


def _load_config(gqe_run: Path):
    cfg_data = json.loads((gqe_run / "config.json").read_text())
    for derived in ("vocab_size", "pool_size"):
        cfg_data.pop(derived, None)
    migration = _migrate_legacy_train(cfg_data.setdefault("train", {}))
    return config_from_dict(cfg_data), migration


def _with_scoring_evaluator(cfg, kind: EvaluatorKind, n_jobs: int):
    """Config clone whose evaluator/n_jobs are the scoring choice, all else identical."""
    train = dataclasses.replace(cfg.train, evaluator=kind, n_jobs=n_jobs)
    return dataclasses.replace(cfg, train=train)


def _score_batches(evaluator, rng, pool_size, seq_len, n_samples, batch_size, on_batch):
    """Draw and score uniform sequences, tracking the running best.

    Calls ``on_batch(n_seen, best_energy, best_idx)`` after each batch so the
    caller can stream a kill-safe trace to disk.
    """
    best_energy = float("inf")
    best_idx = None
    seen = 0
    while seen < n_samples:
        take = min(batch_size, n_samples - seen)
        batch = rng.integers(0, pool_size, size=(take, seq_len), dtype=np.int64)
        final = np.asarray(evaluator(batch))[:, -1]
        i_min = int(np.argmin(final))
        if final[i_min] < best_energy:
            best_energy = float(final[i_min])
            best_idx = batch[i_min].copy()
        seen += take
        on_batch(seen, best_energy, best_idx)
    return best_energy, best_idx


def _refine(evaluator, idx_seq, cfg):
    """Refine one sequence exactly as training._finalize does."""
    pl = cfg.train.pennylane or PennyLaneConfig()
    tokens = [0] + [int(i) + 1 for i in idx_seq]
    t0 = time.perf_counter()
    result = evaluator.refine_tokens(
        tokens,
        qml_device=pl.device,
        shots=pl.refine_shots if pl.refine_shots > 0 else 0,
        seed=cfg.train.seed,
        optimizer=pl.refine_optimizer,
        gradient=pl.refine_gradient,
    )
    return result, tokens, time.perf_counter() - t0


def _verify_scoring(cfg, cfg_scoring, pool, ham, init_angle, n_qubits, rng, n_check):
    """Score the same sequences under mirrored and substituted evaluators."""
    if cfg.train.evaluator is cfg_scoring.train.evaluator:
        return {"performed": False, "reason": "scoring evaluator matches paired run"}
    mirrored, _ = build_evaluator(cfg, pool, ham, init_angle, n_qubits)
    substituted, _ = build_evaluator(cfg_scoring, pool, ham, init_angle, n_qubits)
    batch = rng.integers(0, len(pool), size=(n_check, cfg.train.seq_len), dtype=np.int64)
    a = np.asarray(mirrored(batch))[:, -1]
    b = np.asarray(substituted(batch))[:, -1]
    denom = np.maximum(np.abs(a), 1e-300)
    return {
        "performed": True,
        "n_sequences": int(n_check),
        "mirrored_evaluator": cfg.train.evaluator.value,
        "substituted_evaluator": cfg_scoring.train.evaluator.value,
        "max_abs_diff": float(np.max(np.abs(a - b))),
        "max_rel_diff": float(np.max(np.abs(a - b) / denom)),
    }


def _paired_run_reference(gqe_run: Path):
    """Fall back to the paired run's own metadata.json.

    Needed for paired runs that are not Table 1 cells (the fine-lambda and
    lambda=0.5 extension sweeps), where the canonical CSV has no matching row.
    """
    path = gqe_run / "metadata.json"
    if not path.exists():
        return None
    md = json.loads(path.read_text())
    e0 = md.get("ground_energy")
    if not e0:
        return None
    ref = {"gqe_run": gqe_run.name, "gqe_source_sweep": gqe_run.parent.name, "source": "metadata.json"}
    if md.get("refined_abs_error") is not None:
        ref["gqe_refined_rel_error"] = abs(md["refined_abs_error"]) / abs(e0)
    if md.get("abs_error") is not None:
        ref["gqe_raw_rel_error"] = abs(md["abs_error"]) / abs(e0)
    return ref if "gqe_refined_rel_error" in ref else None


def _canonical_reference(csv_path: Path, pool_variant: str, n_qubits: int, lam: float, seed: int):
    """The paired GQE cell's numbers from the canonical CSV, for side-by-side output."""
    if csv_path is None or not csv_path.exists():
        return None
    import csv as _csv

    with csv_path.open() as fh:
        rows = [r for r in _csv.DictReader(line for line in fh if not line.startswith("#"))]
    for r in rows:
        if (
            r["pool"] == pool_variant
            and int(r["N"]) == n_qubits
            and float(r["lambda"]) == lam
            and int(r["seed"]) == seed
        ):
            return {
                "gqe_refined_rel_error": float(r["refined_rel_error"]),
                "gqe_raw_rel_error": float(r["rel_error"]),
                "gqe_run": r["run"],
                "gqe_source_sweep": r["source_sweep"],
            }
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gqe-run", type=Path, required=True, help="paired GQE run directory")
    ap.add_argument("--draw", type=int, required=True, help="independent draw index (1,2,3,...)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-samples", type=int, default=PRIMARY_BUDGET)
    ap.add_argument(
        "--extra-samples",
        type=int,
        default=EXTRA_BUDGET,
        help="additional samples past the primary budget; best is recorded at both",
    )
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-jobs", type=int, default=16, help="scoring parallelism (speed only)")
    ap.add_argument(
        "--score-evaluator",
        default="auto",
        help="'auto' (D3 fastest admissible), 'mirror', or an EvaluatorKind value",
    )
    ap.add_argument("--refine-device", default=None, help="override; default mirrors paired run")
    ap.add_argument("--verify-scoring", type=int, default=64, help="0 disables the cross-check")
    ap.add_argument("--canonical-csv", type=Path, default=None)
    args = ap.parse_args()

    cfg, migration = _load_config(args.gqe_run)
    n_qubits = cfg.hamiltonian.n_qubits
    lam = cfg.hamiltonian.lam
    seq_len = cfg.train.seq_len
    pool = build_pool(cfg.pool, n_qubits)
    ham = build_hamiltonian(cfg.hamiltonian)
    init_angle = initial_angle_for(cfg)

    if args.refine_device is not None:
        pl = cfg.train.pennylane or PennyLaneConfig()
        cfg = dataclasses.replace(
            cfg,
            train=dataclasses.replace(
                cfg.train, pennylane=dataclasses.replace(pl, device=args.refine_device)
            ),
        )

    if args.score_evaluator == "auto":
        score_kind = _AUTO_SCORER.get(cfg.pool.kind, cfg.train.evaluator)
    elif args.score_evaluator == "mirror":
        score_kind = cfg.train.evaluator
    else:
        score_kind = EvaluatorKind(args.score_evaluator)
    cfg_scoring = _with_scoring_evaluator(cfg, score_kind, args.n_jobs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed = _draw_seed(cfg.pool.kind.value, n_qubits, lam, args.draw)

    print(
        f"=== random-sampler control: {cfg.pool.kind.value} N={n_qubits} lam={lam} "
        f"draw={args.draw} ===\n"
        f"  paired GQE run : {args.gqe_run}\n"
        f"  pool size      : {len(pool)} tokens, L={seq_len}\n"
        f"  scoring        : {score_kind.value} (paired run used "
        f"{cfg.train.evaluator.value}), n_jobs={args.n_jobs}\n"
        f"  refine         : {(cfg.train.pennylane or PennyLaneConfig()).device} "
        f"(mirrored from paired run)\n"
        f"  rng seed       : {seed}"
    )
    if migration is not None:
        print(f"  legacy schema  : migrated {sorted(migration['legacy_keys'])} -> train.pennylane")

    check = (
        _verify_scoring(
            cfg,
            cfg_scoring,
            pool,
            ham,
            init_angle,
            n_qubits,
            np.random.default_rng(seed ^ 0xA5A5),
            args.verify_scoring,
        )
        if args.verify_scoring > 0
        else {"performed": False, "reason": "disabled"}
    )
    if check["performed"]:
        print(
            f"  scoring check  : max rel diff {check['max_rel_diff']:.2e} "
            f"over {check['n_sequences']} sequences"
        )

    evaluator, ground_energy = build_evaluator(cfg_scoring, pool, ham, init_angle, n_qubits)

    trace = args.out_dir / "trace.csv"
    trace.write_text("n_samples,best_raw_energy,best_raw_rel_error\n")
    fh = trace.open("a")

    def on_batch(seen, best_energy, _best_idx):
        rel = abs(best_energy - ground_energy) / abs(ground_energy)
        fh.write(f"{seen},{best_energy:.12g},{rel:.6e}\n")
        fh.flush()

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    best_energy, best_idx = _score_batches(
        evaluator, rng, len(pool), seq_len, args.n_samples, args.batch_size, on_batch
    )
    primary_seconds = time.perf_counter() - t0
    primary = {"n_samples": args.n_samples, "raw_energy": best_energy, "idx": best_idx}
    print(
        f"  best raw @{args.n_samples}: {best_energy:.10f} "
        f"(exact {ground_energy:.10f}, {primary_seconds:.1f}s)"
    )

    extended = None
    if args.extra_samples > 0:
        e2, i2 = _score_batches(
            evaluator,
            rng,
            len(pool),
            seq_len,
            args.extra_samples,
            args.batch_size,
            lambda seen, be, bi: on_batch(args.n_samples + seen, min(be, best_energy), bi),
        )
        if e2 < best_energy:
            extended = {"n_samples": args.n_samples + args.extra_samples, "raw_energy": e2, "idx": i2}
            print(f"  best raw @{extended['n_samples']}: {e2:.10f} (improved)")
        else:
            print(f"  best raw @{args.n_samples + args.extra_samples}: unchanged")
    fh.close()

    refined, tokens, refine_seconds = _refine(evaluator, primary["idx"], cfg)
    rel_refined = abs(refined.energy - ground_energy) / abs(ground_energy)
    rel_raw = abs(primary["raw_energy"] - ground_energy) / abs(ground_energy)
    print(
        f"  refined @{args.n_samples}: {primary['raw_energy']:.6f} -> {refined.energy:.10f} "
        f"(rel err {rel_refined:.3e}, {refined.n_evaluations} evals, {refine_seconds:.1f}s)"
    )

    extended_record = None
    if extended is not None:
        r2, tokens2, secs2 = _refine(evaluator, extended["idx"], cfg)
        rel2 = abs(r2.energy - ground_energy) / abs(ground_energy)
        extended_record = {
            "n_samples": extended["n_samples"],
            "raw_energy": extended["raw_energy"],
            "raw_rel_error": abs(extended["raw_energy"] - ground_energy) / abs(ground_energy),
            "refined_energy": r2.energy,
            "refined_rel_error": rel2,
            "tokens": tokens2,
            "refine_seconds": secs2,
        }
        print(f"  refined @{extended['n_samples']}: rel err {rel2:.3e}")

    pool_variant = args.gqe_run.name.split("_n_qubits")[0]
    reference = _canonical_reference(
        args.canonical_csv, pool_variant, n_qubits, lam, cfg.train.seed
    ) or _paired_run_reference(args.gqe_run)
    if reference:
        ratio = reference["gqe_refined_rel_error"] / rel_refined if rel_refined > 0 else float("inf")
        print(
            f"  paired GQE cell: refined rel err {reference['gqe_refined_rel_error']:.3e} "
            f"-> random/GQE = {1 / ratio:.3g}x"
            if ratio
            else ""
        )

    metadata = {
        "control": "compute-matched uniform random sampler",
        "pool_variant": pool_variant,
        "pool_kind": cfg.pool.kind.value,
        "pool_size": len(pool),
        "n_qubits": n_qubits,
        "lam": lam,
        "seq_len": seq_len,
        "draw": args.draw,
        "rng_seed": seed,
        "paired_gqe_run": str(args.gqe_run),
        "paired_gqe_seed": cfg.train.seed,
        "ground_energy": ground_energy,
        "scoring_evaluator": score_kind.value,
        "paired_run_evaluator": cfg.train.evaluator.value,
        "legacy_config_migration": migration,
        "scoring_cross_check": check,
        "refine_device": (cfg.train.pennylane or PennyLaneConfig()).device,
        "primary": {
            "n_samples": args.n_samples,
            "raw_energy": primary["raw_energy"],
            "raw_rel_error": rel_raw,
            "refined_energy": refined.energy,
            "refined_rel_error": rel_refined,
            "refine_evaluations": refined.n_evaluations,
            "refine_seconds": refine_seconds,
            "scoring_seconds": primary_seconds,
        },
        "extended": extended_record,
        "canonical_reference": reference,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.out_dir / "best_sequence.json").write_text(
        json.dumps(
            {
                "tokens": tokens,
                "op_labels": [pool[t - 1].label for t in tokens[1:]],
                "energy": primary["raw_energy"],
                "refined_energy": refined.energy,
                "refined_angles": refined.angles,
                "ground_energy": ground_energy,
            },
            indent=2,
        )
        + "\n"
    )
    (args.out_dir / "config.json").write_text(
        json.dumps(
            {
                **config_to_dict(cfg),
                "vocab_size": len(pool) + 1,
                "pool_size": len(pool),
                "_control_overrides": {
                    "sampler": "uniform random over pool indices",
                    "scoring_evaluator": score_kind.value,
                    "scoring_n_jobs": args.n_jobs,
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"  wrote {args.out_dir}")


if __name__ == "__main__":
    main()
