"""Collect sweep run directories into one results table."""

import json
import math
from pathlib import Path

import pandas as pd

from spingqe_lmg.analysis.plots import _EXTENDED_PAULI_WORDS
from spingqe_lmg.enums import Connectivity, PoolKind, Variant

_BASE_ANGLE = math.pi / 2  # DEFAULT_ANGLES[0] in pools.py


def _effective_angle_scale(pool_cfg: dict, n_qubits: int) -> float:
    """Recover the vocabulary-rescaling factor, robust to config vintage.

    Two config.json vintages coexist in `runs/`. Current configs serialize
    an explicit ``pool.angle_scale`` field and store the *pre*-rescaling
    angles (the transform runs at ``build_pool()`` time, not before save) —
    for these, trust the field. Older configs predate that field entirely
    (the key is absent, not 0) and instead baked the already-rescaled
    absolute angles directly into ``pool.angles`` — for these, the field is
    missing and the scale must be recovered from the resolved first angle:
    unscaled runs have ``angles[0] == pi/2`` exactly, and a rescaled run's
    first angle is ``(pi/2) * angle_scale / n_qubits``.
    """
    explicit = pool_cfg.get("angle_scale")
    if explicit is not None:
        return float(explicit)
    angles = pool_cfg.get("angles")
    if not angles:
        return 0.0
    first = angles[0]
    if math.isclose(first, _BASE_ANGLE, rel_tol=1e-9):
        return 0.0
    return round(first * n_qubits / _BASE_ANGLE, 6)


def _compute_variant(
    pool_kind: str,
    init_state: str,
    paulis: str,
    connectivity: str,
) -> str | None:
    """Derive the canonical variant label from pool and reference-state config."""
    if pool_kind == PoolKind.COLLECTIVE.value:
        return Variant.COLLECTIVE.value
    if connectivity == Connectivity.NN.value:
        return Variant.PAIRWISE_NN.value
    if pool_kind == PoolKind.PAULI_PAIR.value:
        if bool(set(paulis.split(",") if paulis else ()) & _EXTENDED_PAULI_WORDS):
            return Variant.PAIRWISE_EXT_MF.value
        return Variant.PAIRWISE_ALL.value
    return None


def aggregate_sweep(sweep_dir: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Walk run dirs under ``sweep_dir``; return (results table, incomplete run names).

    A run is complete when metadata.json exists (written at the end of
    training). The table carries the physics parameters from config.json so
    QPT curves can be grouped directly.  A ``variant`` column is computed
    from pool config (not run-name heuristics) and uses the canonical
    ``Variant`` enum values.
    """
    sweep_dir = Path(sweep_dir)
    rows = []
    incomplete = []
    for run_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        meta_path = run_dir / "metadata.json"
        cfg_path = run_dir / "config.json"
        if not (meta_path.exists() and cfg_path.exists()):
            if (run_dir / "losses.csv").exists() or (run_dir / "checkpoints").exists():
                incomplete.append(run_dir.name)
            continue
        meta = json.loads(meta_path.read_text())
        cfg = json.loads(cfg_path.read_text())
        ham = cfg["hamiltonian"]
        pool_cfg = cfg.get("pool", {})
        paulis_raw = pool_cfg.get("paulis", ())
        init_state = cfg["train"].get("init_state", meta.get("init_state", "zero"))
        row = {
            "run": run_dir.name,
            "kind": ham["kind"],
            "n_qubits": ham["n_qubits"],
            "h": ham["h"],
            "lam": ham["lam"],
            "gamma": ham["gamma"],
            "pool_kind": pool_cfg["kind"],
            "connectivity": pool_cfg["connectivity"],
            "seed": cfg["train"]["seed"],
            "init_state": init_state,
            "angle_scale": _effective_angle_scale(pool_cfg, ham["n_qubits"]),
            "paulis": ",".join(paulis_raw) if paulis_raw else "",
            "pool_size": cfg.get("pool_size"),
            "seq_len": cfg["train"].get("seq_len"),
            "ground_energy": meta["ground_energy"],
            "best_energy": meta["best_energy"],
            "best_epoch": meta["best_epoch"],
            "abs_error": meta["abs_error"],
            "refined_energy": meta.get("refined_energy"),
            "refined_abs_error": meta.get("refined_abs_error"),
        }
        scale = abs(meta["ground_energy"]) or 1.0
        row["rel_error"] = row["abs_error"] / scale
        if row["refined_abs_error"] is not None:
            row["refined_rel_error"] = row["refined_abs_error"] / scale
        row["variant"] = _compute_variant(
            pool_cfg["kind"],
            init_state,
            ",".join(paulis_raw) if paulis_raw else "",
            pool_cfg["connectivity"],
        )
        rows.append(row)
    return pd.DataFrame(rows), incomplete
