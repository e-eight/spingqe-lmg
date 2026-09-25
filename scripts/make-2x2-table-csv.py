#!/usr/bin/env python3
"""Emit the data behind Table VII as a CSV, at ten seeds and as medians.

Every row is generated through one code path, as a ten-seed median. (An
earlier hand-filled version of two upper-block rows printed the seed minimum
instead of the median.)

Two sources, because the four pool-reference combinations do not all live in
one sweep: the canonical pool-accuracy CSV supplies the zero-reference
standard pool and the mean-field extended pool at N=16, and the
pool-vs-reference sweep supplies the two cells that completed the 2x2.

2026-07-30: the lower block now covers all four pairwise configurations, not
just the two the pool-vs-reference sweep happened to run. Only that sweep
carries per-seed reference AND refined errors together, so the other two rows
take the refined error from the canonical CSV and the reference error from run
metadata, which is what the collective rows already did. The untouched
reference state's error depends on (N, lambda, reference) and not on the pool,
and the two sweeps agree on it to full precision, so the pairing is exact
rather than approximate.

Run from the repository root; inputs and output default to data/.
"""

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CANONICAL_CSV = REPO / "data" / "tables" / "pool-accuracy-l12-s1-10-2026-07-29.csv"
PVR_ROOT = REPO / "data" / "runs" / "pool-vs-reference-2026-07-26"
# Campaign roots are pinned by name: several unrelated sweeps under
# data/runs/ contain run directories with identical names, so globbing
# for a run name would silently pick the wrong campaign.
RUNS_ROOT = PVR_ROOT.parent
PAIRWISE_ROOT = RUNS_ROOT / "pool-accuracy-pairwise-l12-s1-10-2026-07-29"
COLLECTIVE_ROOT = RUNS_ROOT / "pool-accuracy-l12-s1-10-2026-07-28"
COLLECTIVE_MF_ROOT = RUNS_ROOT / "collective-mf-n16-2026-07-30"
OUT_CSV = REPO / "data" / "tables" / "2x2-table-2026-07-29.csv"
N_SEEDS = 10
N_QUBITS = 16


def median(values):
    """Linear-interpolation median, matching the canonical generator's pct(50)."""
    xs = sorted(values)
    n = len(xs)
    if n == 0:
        raise ValueError("median of an empty sequence")
    mid = (n - 1) / 2
    lo, hi = int(mid), int(mid) + (0 if mid == int(mid) else 1)
    if lo == hi:
        return xs[lo]
    return (xs[lo] + xs[hi]) / 2


def gain(reference, refined):
    """Factor by which refinement improved on the untouched reference state."""
    if refined == 0:
        raise ValueError("refined error is zero; gain is undefined")
    return reference / refined


def _reference_error(pool, lam, root):
    """Untouched reference-state relative error, from one run's metadata.

    The reference state is fixed by (N, lambda, reference) and is identical
    across seeds, so seed 1 suffices. Used for the two lower-block rows whose
    refined error comes from the canonical CSV, which carries no reference
    column.
    """
    run = root / f"{pool}_n_qubits{N_QUBITS}_lam{lam}_s1"
    meta = run / "metadata.json"
    if not meta.exists():
        raise FileNotFoundError(f"no metadata for {run}")
    m = json.loads(meta.read_text())
    return abs(m["initial_energy"] - m["ground_energy"]) / (abs(m["ground_energy"]) or 1.0)


def check_seed_count(label, n_seeds):
    if n_seeds != N_SEEDS:
        raise ValueError(f"{label}: expected {N_SEEDS} seeds, found {n_seeds}")


def _from_canonical(pool, lam):
    """Per-seed refined relative errors for one cell of the canonical CSV."""
    with open(CANONICAL_CSV) as f:
        rows = [
            r
            for r in csv.DictReader(line for line in f if not line.startswith("#"))
            if r["pool"] == pool
            and int(r["N"]) == N_QUBITS
            and float(r["lambda"]) == lam
            and r["refined_rel_error"]
        ]
    check_seed_count(f"{pool} lam={lam}", len(rows))
    return [float(r["refined_rel_error"]) for r in rows]


def _from_pvr(variant, lam):
    """Per-seed (reference, refined) relative errors from the 2x2 sweep."""
    ref, refined = [], []
    for seed in range(1, N_SEEDS + 1):
        meta = PVR_ROOT / f"{variant}_lam{lam}_s{seed}" / "metadata.json"
        if not meta.exists():
            continue
        m = json.loads(meta.read_text())
        scale = abs(m["ground_energy"]) or 1.0
        ref.append(abs(m["initial_energy"] - m["ground_energy"]) / scale)
        refined.append(m["refined_abs_error"] / scale)
    check_seed_count(f"{variant} lam={lam}", len(refined))
    return ref, refined


def _from_collective_mf(lam):
    """Per-seed (reference, refined) errors for collective + mean field at N=16.

    Its own campaign (2026-07-30), run to close the one configuration the gain block
    could not otherwise cover: every other row uses the zero reference, and no
    collective + mean_field run existed below N=50.
    """
    ref, refined = [], []
    for seed in range(1, N_SEEDS + 1):
        meta = (COLLECTIVE_MF_ROOT
                / f"collective-mf_n_qubits{N_QUBITS}_lam{lam}_s{seed}"
                / "metadata.json")
        if not meta.exists():
            continue
        m = json.loads(meta.read_text())
        scale = abs(m["ground_energy"]) or 1.0
        ref.append(abs(m["initial_energy"] - m["ground_energy"]) / scale)
        refined.append(m["refined_abs_error"] / scale)
    check_seed_count(f"collective-mf lam={lam}", len(refined))
    return ref, refined


def main():
    out = []

    # Upper block: the four pool-reference combinations. Pools are named rather
    # than spelled out as Pauli-word lists, matching the paper's Sec. III
    # terminology and the lower block's labels; Sec. VII's prose names YZ and XY
    # where the distinction matters.
    for pool, reference, getter in (
        ("std.\\ pair.", "zero", lambda lam: _from_canonical("pairwise-all", lam)),
        ("std.\\ pair.", "mean field", lambda lam: _from_pvr("pairwise-all-mf", lam)[1]),
        ("ext.\\ pair.", "zero", lambda lam: _from_pvr("pairwise-ext-zero", lam)[1]),
        ("ext.\\ pair.", "mean field", lambda lam: _from_canonical("pairwise-ext-mf", lam)),
    ):
        for lam in (1.5, 2.0):
            out.append({
                "block": "upper",
                "pool": pool,
                "reference": reference,
                "configuration": "",
                "lam": lam,
                "reference_rel_error": "",
                "refined_rel_error": median(getter(lam)),
                "gain": "",
                "n_seeds": N_SEEDS,
            })

    # Lower block: refined error against each configuration's own reference,
    # for all four pairwise configurations. The pool-vs-reference sweep gives
    # both errors per seed for two of them; the other two take the refined
    # error from the canonical CSV and the reference error from run metadata.
    def _pvr_pair(variant, lam):
        return tuple(median(v) for v in _from_pvr(variant, lam))

    for pool, reference, lam, r, f_ in (
        ("std.\\ pair.", "zero", 2.0,
         _reference_error("pairwise-all", 2.0, PAIRWISE_ROOT),
         median(_from_canonical("pairwise-all", 2.0))),
        ("std.\\ pair.", "mean field", 2.0, *_pvr_pair("pairwise-all-mf", 2.0)),
        ("ext.\\ pair.", "zero", 2.0, *_pvr_pair("pairwise-ext-zero", 2.0)),
        ("ext.\\ pair.", "mean field", 2.0,
         _reference_error("pairwise-ext-mf", 2.0, PAIRWISE_ROOT),
         median(_from_canonical("pairwise-ext-mf", 2.0))),
    ):
        out.append({
            "block": "lower", "pool": pool, "reference": reference,
            "configuration": f"{pool}, {reference}", "lam": lam,
            "reference_rel_error": r, "refined_rel_error": f_,
            "gain": gain(r, f_), "n_seeds": N_SEEDS,
        })

    # Lower block, collective rows: reference error is the zero-state error,
    # identical across seeds, so read it from any one run's metadata.
    for lam in (2.0, 1.5):
        r = _reference_error("collective", lam, COLLECTIVE_ROOT)
        f_ = median(_from_canonical("collective", lam))
        out.append({
            "block": "lower", "pool": "collective", "reference": "zero",
            "configuration": "collective, zero", "lam": lam,
            "reference_rel_error": r, "refined_rel_error": f_,
            "gain": gain(r, f_), "n_seeds": N_SEEDS,
        })

    # Lower block, collective + mean field. Both couplings are emitted for the
    # record; the paper quotes lambda=2.0 only.
    for lam in (2.0, 1.5):
        r, f_ = (median(v) for v in _from_collective_mf(lam))
        out.append({
            "block": "lower", "pool": "collective", "reference": "mean field",
            "configuration": "collective, mean field", "lam": lam,
            "reference_rel_error": r, "refined_rel_error": f_,
            "gain": gain(r, f_), "n_seeds": N_SEEDS,
        })

    fields = ["block", "pool", "reference", "configuration", "lam",
              "reference_rel_error", "refined_rel_error", "gain", "n_seeds"]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        f.write(
            "# Table VII (upper block) and the pool-reference gain (lower block), "
            "all rows medians over seeds 1-10. Generated by "
            "scripts/make-2x2-table-csv.py.\n"
        )
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {OUT_CSV} with {len(out)} rows")


if __name__ == "__main__":
    main()
