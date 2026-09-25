"""Drive the random-sampler control over the pool-accuracy cells named on the command line.

Cells come from the 3-seed pool-accuracy CSV
(data/tables/pool-accuracy-l12-s123-2026-07-25.csv). Each draw is matched
config-for-config against the GQE run archived for that cell at the time
this control ran: draw d against seed d for the same (pool, N, lambda).
That pairing is what makes the comparison valid, and it is internal to this
campaign. It is NOT a claim that the paired runs are the ones the
manuscript's current table reports: the 2026-07-29 pairwise campaign
re-measured 28 of those cells at ten seeds, so for 84 of the 102 pairwise
pairs the table now reports a different run of the same cell.

Nothing is hardcoded here; changing the grid means changing the filters.

Writes one directory per cell plus a summary CSV streamed row-by-row, so a
killed job still leaves a usable partial comparison.

Usage:
    python scripts/random-sampler-control-grid.py \
        --canonical-csv data/tables/pool-accuracy-l12-s123-2026-07-25.csv \
        --runs-root <runs-root holding the paired GQE runs> \
        --out-root  <runs-root>/random-sampler-control-2026-07-25 \
        --n-qubits 16 --lambdas 1.5,2.0
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent / "random-sampler-control.py"

SUMMARY_HEADER = (
    "pool,N,lambda,draw,paired_seed,n_samples,"
    "random_raw_rel_error,random_refined_rel_error,"
    "gqe_raw_rel_error,gqe_refined_rel_error,random_over_gqe,"
    "extended_refined_rel_error,scoring_max_rel_diff,paired_run\n"
)


def load_cells_from_sweep(sweep_dir: Path, runs_root: Path):
    """Enumerate cells directly from a sweep's run directories.

    Used for the fine-lambda and lambda=0.5 extension sweeps, whose cells are not
    rows of the pool-accuracy grid and so do not appear in the canonical CSV.  Reads (N, lambda,
    seed) from each run's config.json rather than parsing directory names.
    """
    cells = []
    for run_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        ham, train = cfg.get("hamiltonian", {}), cfg.get("train", {})
        cells.append(
            {
                "pool": run_dir.name.split("_n_qubits")[0],
                "N": str(ham.get("n_qubits")),
                "lambda": str(ham.get("lam")),
                "seed": str(train.get("seed")),
                "run": run_dir.name,
                "source_sweep": str(sweep_dir.relative_to(runs_root)),
            }
        )
    cells.sort(key=lambda r: (r["pool"], int(r["N"]), float(r["lambda"]), int(r["seed"])))
    return cells


def load_cells(csv_path: Path, n_qubits: list[int], lambdas: list[float], pools: list[str] | None):
    with csv_path.open() as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    cells = [
        r
        for r in rows
        if int(r["N"]) in n_qubits
        and float(r["lambda"]) in lambdas
        and (pools is None or r["pool"] in pools)
    ]
    cells.sort(key=lambda r: (r["pool"], int(r["N"]), float(r["lambda"]), int(r["seed"])))
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical-csv", type=Path, required=True)
    ap.add_argument(
        "--from-sweep",
        type=Path,
        default=None,
        help="enumerate cells from this sweep's run dirs instead of the canonical CSV "
        "(for sweeps whose cells are not pool-accuracy grid rows); --canonical-csv is "
        "still passed through so any cell that IS a grid row gets its reference numbers",
    )
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--n-qubits", default="16", help="comma-separated system sizes")
    ap.add_argument("--lambdas", default="1.5,2.0")
    ap.add_argument("--pools", default=None, help="comma-separated; default all in the CSV")
    ap.add_argument("--n-samples", type=int, default=7010)
    ap.add_argument("--extra-samples", type=int, default=1400)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument(
        "--refine-device",
        default=None,
        help="uniform refinement device; default mirrors each paired run. "
        "Refinement is backend-neutral to 9.7e-16 (2026-07-26 measurement), so a "
        "uniform override removes a confound from the control at no cost.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    lambdas = [float(x) for x in args.lambdas.split(",")]
    n_qubits_list = [int(x) for x in args.n_qubits.split(",")]
    pools = args.pools.split(",") if args.pools else None
    if args.from_sweep:
        cells = load_cells_from_sweep(args.from_sweep, args.runs_root)
    else:
        cells = load_cells(args.canonical_csv, n_qubits_list, lambdas, pools)
    if not cells:
        sys.exit("no cells matched the filters")

    args.out_root.mkdir(parents=True, exist_ok=True)
    summary = args.out_root / "summary.csv"
    if not summary.exists():
        summary.write_text(SUMMARY_HEADER)

    print(f"{len(cells)} cells to run into {args.out_root}\n")
    failures = []

    for i, cell in enumerate(cells, 1):
        pool, lam, seed = cell["pool"], float(cell["lambda"]), int(cell["seed"])
        n_qubits = int(cell["N"])
        paired = args.runs_root / cell["source_sweep"] / cell["run"]
        # Draw index tracks the paired GQE seed so best-of-3 and median-of-3
        # are computed over matched sets on both sides.
        draw = seed
        out_dir = args.out_root / f"{pool}_n_qubits{n_qubits}_lam{lam}_draw{draw}"
        label = f"[{i}/{len(cells)}] {pool} N={n_qubits} lam={lam} draw={draw}"

        if not (paired / "config.json").exists():
            print(f"{label}: SKIP (paired run missing: {paired})")
            failures.append((label, "paired run missing"))
            continue
        if (out_dir / "metadata.json").exists():
            print(f"{label}: already done, skipping")
            continue

        cmd = [
            sys.executable,
            str(SCRIPT),
            "--gqe-run", str(paired),
            "--draw", str(draw),
            "--out-dir", str(out_dir),
            "--n-samples", str(args.n_samples),
            "--extra-samples", str(args.extra_samples),
            "--n-jobs", str(args.n_jobs),
            "--canonical-csv", str(args.canonical_csv),
        ]
        if args.refine_device:
            cmd += ["--refine-device", args.refine_device]
        print(f"{label}")
        if args.dry_run:
            print("  " + " ".join(cmd))
            continue

        proc = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write("".join(f"  {ln}\n" for ln in proc.stdout.strip().splitlines()))
        if proc.returncode != 0:
            print(f"  FAILED (rc={proc.returncode})")
            sys.stderr.write(proc.stderr[-2000:])
            failures.append((label, f"rc={proc.returncode}"))
            continue

        md = json.loads((out_dir / "metadata.json").read_text())
        p, ref = md["primary"], md.get("canonical_reference") or {}
        ext = md.get("extended") or {}
        gqe_refined = ref.get("gqe_refined_rel_error")
        ratio = (
            p["refined_rel_error"] / gqe_refined
            if gqe_refined not in (None, 0)
            else ""
        )
        check = md.get("scoring_cross_check") or {}
        with summary.open("a") as fh:
            fh.write(
                f"{pool},{n_qubits},{lam},{draw},{seed},{p['n_samples']},"
                f"{p['raw_rel_error']:.6e},{p['refined_rel_error']:.6e},"
                f"{ref.get('gqe_raw_rel_error', '')},{gqe_refined if gqe_refined is not None else ''},"
                f"{ratio if ratio == '' else format(ratio, '.4g')},"
                f"{format(ext['refined_rel_error'], '.6e') if ext else ''},"
                f"{format(check['max_rel_diff'], '.2e') if check.get('performed') else ''},"
                f"{cell['run']}\n"
            )

    print(f"\nsummary: {summary}")
    if failures:
        print(f"{len(failures)} cell(s) did not complete:")
        for label, why in failures:
            print(f"  {label}: {why}")
        sys.exit(1)


if __name__ == "__main__":
    main()
