#!/usr/bin/env python3
"""Build the canonical pool-accuracy 10-seed CSV over the full 51-cell grid.

Every row comes from a 10-seed campaign at seq_len=12: the 2026-07-28
collective campaign owns the 17 collective cells and six pairwise cells, and
the 2026-07-29 pairwise campaign owns the other 28 pairwise cells. There is
no legacy-sweep merging and no three-seed fill -- only 10-seed campaign runs
on one uniform configuration.

Emits one row per (pool, N, lambda, seed) plus per-cell summary columns
(median_rel_err, iqr_dex, p10, p90, min, max, k_within_10x, spread) repeated
on every row of that cell, so a single CSV serves both the per-seed audit
and the table-generation script without a second file.

Run from the repository root; inputs and output default to data/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO / "data" / "runs"
# Ordered: cells resolve to the FIRST sweep that contains them. The 07-28
# campaign owns the collective grid and six pairwise cells; the 07-29
# campaign owns the other 28 pairwise cells. The two do not overlap, so the
# order is documentation rather than a tie-break, but keep it explicit.
SWEEPS = (
    "pool-accuracy-l12-s1-10-2026-07-28",
    "pool-accuracy-pairwise-l12-s1-10-2026-07-29",
)
OUT_CSV = REPO / "data" / "tables" / "pool-accuracy-l12-s1-10-2026-07-29.csv"
SEEDS = tuple(range(1, 11))
L = 12  # seq_len; every input row must have config.json train.seq_len == 12

# 51 cells = the full grid. Collective: N in {8,10,12,14,16} x lam in
# {1.0,1.5,2.0}, plus lam=0.5 at N in {8,10} (collective-main only sweeps
# lam in {1,1.5,2}; smalln-lam05 covers lam=0.5 for N in {8,10}) = 17.
# Pairwise: both pools x the same 17 (N, lam) combinations = 34.
# 17 + 34 = 51, which is exactly the grid the random-sampler control spans.
CELLS = set()
_NL_COMBOS = {(n, lam) for n in (8, 10, 12, 14, 16) for lam in (1.0, 1.5, 2.0)}
_NL_COMBOS |= {(n, 0.5) for n in (8, 10)}
assert len(_NL_COMBOS) == 17, len(_NL_COMBOS)
for _n, _lam in _NL_COMBOS:
    CELLS.add(("collective", _n, _lam))
    for _pool in ("pairwise-all", "pairwise-ext-mf"):
        CELLS.add((_pool, _n, _lam))
assert len(CELLS) == 51, len(CELLS)


def _find_run_dir(sweep_dir: Path, pool: str, n: int, lam: float, seed: int) -> Path | None:
    run_id = f"{pool}_n_qubits{n}_lam{lam}_s{seed}"
    d = sweep_dir / run_id
    return d if (d / "metadata.json").exists() and (d / "config.json").exists() else None


def _config_sha256(run_dir: Path) -> str:
    return hashlib.sha256((run_dir / "config.json").read_bytes()).hexdigest()


def _load_row(run_dir: Path, pool: str, n: int, lam: float, seed: int, sweep: str):
    cfg = json.loads((run_dir / "config.json").read_text())
    if cfg["train"]["seq_len"] != L:
        return None
    meta = json.loads((run_dir / "metadata.json").read_text())
    ground_energy = meta["ground_energy"]
    scale = abs(ground_energy) or 1.0
    abs_error = meta["abs_error"]
    refined_abs_error = meta.get("refined_abs_error")
    return {
        "pool": pool,
        "N": n,
        "lambda": lam,
        "seed": seed,
        "refined_rel_error": (refined_abs_error / scale if refined_abs_error is not None else ""),
        "rel_error": abs_error / scale,
        "run": run_dir.name,
        "source_sweep": sweep,
        "config_sha256": _config_sha256(run_dir),
        "seq_len": cfg["train"]["seq_len"],
    }


def _cell_stats(errs: list[float]) -> dict:
    xs = sorted(errs)
    n = len(xs)

    def pct(p):
        # linear-interpolation percentile, matching numpy's default
        idx = p / 100 * (n - 1)
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return xs[lo]
        frac = idx - lo
        return xs[lo] * (1 - frac) + xs[hi] * frac

    median = pct(50)
    q1 = pct(25)
    q3 = pct(75)
    iqr_dex = math.log10(q3) - math.log10(q1) if q1 > 0 else float("nan")
    lo, hi = xs[0], xs[-1]
    k_within_10x = sum(1 for x in xs if x <= lo * 10)
    spread = hi / lo if lo > 0 else float("inf")
    return {
        "median_rel_err": median,
        "iqr_dex": iqr_dex,
        "p10": pct(10),
        "p90": pct(90),
        "min": lo,
        "max": hi,
        "k_within_10x": k_within_10x,
        "spread": spread,
    }


def main():
    rows = []
    errors = []

    for pool, n, lam in sorted(CELLS):
        for seed in SEEDS:
            found = None
            for sweep in SWEEPS:
                d = _find_run_dir(RUNS_ROOT / sweep, pool, n, lam, seed)
                if d is not None:
                    found = (sweep, d)
                    break
            if found is None:
                errors.append(
                    f"NO MATCH in {list(SWEEPS)}: pool={pool} N={n} lam={lam} seed={seed}"
                )
                continue
            sweep, d = found
            row = _load_row(d, pool, n, lam, seed, sweep)
            if row is None:
                errors.append(f"BAD seq_len: pool={pool} N={n} lam={lam} seed={seed} in {d}")
                continue
            rows.append(row)

    if errors:
        raise SystemExit(
            f"make-pool-accuracy-10seed-csv: {len(errors)} cell(s) failed:\n"
            + "\n".join(errors)
        )

    keys = [(r["pool"], r["N"], r["lambda"], r["seed"]) for r in rows]
    if len(keys) != len(set(keys)):
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        raise SystemExit(f"make-pool-accuracy-10seed-csv: duplicate rows: {dupes}")

    expected_n_rows = len(CELLS) * len(SEEDS)
    if len(rows) != expected_n_rows:
        raise SystemExit(f"expected {expected_n_rows} rows, got {len(rows)}")

    # per-cell summary stats over refined_rel_error
    by_cell: dict[tuple, list[float]] = {}
    for r in rows:
        cell = (r["pool"], r["N"], r["lambda"])
        by_cell.setdefault(cell, []).append(r["refined_rel_error"])

    stats_by_cell = {cell: _cell_stats(errs) for cell, errs in by_cell.items()}
    for r in rows:
        cell = (r["pool"], r["N"], r["lambda"])
        r.update(stats_by_cell[cell])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pool",
        "N",
        "lambda",
        "seed",
        "refined_rel_error",
        "rel_error",
        "run",
        "source_sweep",
        "config_sha256",
        "seq_len",
        "median_rel_err",
        "iqr_dex",
        "p10",
        "p90",
        "min",
        "max",
        "k_within_10x",
        "spread",
    ]
    rows.sort(key=lambda r: (r["pool"], r["N"], r["lambda"], r["seed"]))
    with open(OUT_CSV, "w", newline="") as f:
        f.write(
            "# Canonical pool-accuracy 10-seed CSV over the full 51-cell grid "
            "(17 collective + 34 pairwise), generated by "
            "scripts/make-pool-accuracy-10seed-csv.py. "
            "seeds 1-10, seq_len=12. Per-cell summary columns "
            "(median_rel_err, iqr_dex, p10, p90, min, max, k_within_10x, "
            "spread) repeated on every row of that cell.\n"
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_CSV} with {len(rows)} rows ({expected_n_rows} expected)")


if __name__ == "__main__":
    main()
