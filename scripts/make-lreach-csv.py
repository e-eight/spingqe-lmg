"""Aggregate a reachability-depth (onset) scan into one canonical CSV.

Reads every per-cell CSV under a sweep directory's `cells/` subdirectory
(written by `scripts/reachability-depth-scan.py`, run as a Slurm array,
one file per (pool, N, lambda, L) manifest row) and aggregates to one row per
cell: best-of-R relative error, success fraction (restarts reaching the
1e-10 floor), and median per-restart seconds. Asserts the expected restart
count per cell so a partial cell fails loudly rather than silently
understating coverage.

Usage:
    python scripts/make-lreach-csv.py \
        --sweep-dir data/runs/reachability-depth-grid-2026-07-26 \
        --out data/tables/lreach-2026-07-26.csv \
        --expected-restarts 12
"""

import argparse
import csv
import statistics
from pathlib import Path

FIELDS = (
    "pool",
    "n_qubits",
    "lam",
    "seq_len",
    "n_restarts",
    "best_rel_error",
    "success_fraction",
    "median_seconds",
)


def aggregate(cells_dir: Path, expected_restarts: int) -> list[dict]:
    rows = []
    for cell_path in sorted(cells_dir.glob("cell-*.csv")):
        with open(cell_path) as f:
            cell_rows = list(csv.DictReader(f))
        if not cell_rows:
            raise ValueError(f"{cell_path} has no data rows")
        n = len(cell_rows)
        if n != expected_restarts:
            raise ValueError(
                f"{cell_path} has {n} restarts, expected {expected_restarts} "
                "(a partial cell must be rerun, not silently aggregated)"
            )
        pool = cell_rows[0]["pool"]
        n_qubits = cell_rows[0]["n_qubits"]
        lam = cell_rows[0]["lam"]
        seq_len = cell_rows[0]["seq_len"]
        rel_errors = [float(r["rel_error"]) for r in cell_rows]
        seconds = [float(r["seconds"]) for r in cell_rows]
        reached = [r["reached"].strip().lower() == "true" for r in cell_rows]
        rows.append(
            {
                "pool": pool,
                "n_qubits": int(n_qubits),
                "lam": float(lam),
                "seq_len": int(seq_len),
                "n_restarts": n,
                "best_rel_error": min(rel_errors),
                "success_fraction": sum(reached) / n,
                "median_seconds": statistics.median(seconds),
            }
        )
    rows.sort(key=lambda r: (r["pool"], r["n_qubits"], r["lam"], r["seq_len"]))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expected-restarts", type=int, default=12)
    args = p.parse_args()

    rows = aggregate(args.sweep_dir / "cells", args.expected_restarts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        f.write(
            "# Aggregated by scripts/make-lreach-csv.py from "
            f"{args.sweep_dir}/cells/\n"
        )
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
