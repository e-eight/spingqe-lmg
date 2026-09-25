"""Console entry points: spingqe-exact, spingqe-sweep-plan, spingqe-aggregate, spingqe-state-quality."""

import argparse
import csv
import itertools
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from spingqe_lmg.analysis.aggregate import aggregate_sweep
from spingqe_lmg.analysis.plots import plot_exact_curves, plot_qpt_errors
from spingqe_lmg.analysis.state_quality import analyze_sweep
from spingqe_lmg.config import (
    MANIFEST_FIXED_COLUMNS,
    config_from_dict,
    load_config,
    manifest_row_to_overrides,
    toml_literal,
)
from spingqe_lmg.exact import exact_qpt_curve


def exact_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Exact LMG QPT curves (Dicke sector).")
    parser.add_argument("--n", type=int, nargs="+", default=[4, 6, 8, 10, 100, 1000])
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--lam-min", type=float, default=0.0)
    parser.add_argument("--lam-max", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=81)
    parser.add_argument("--out-dir", default=".", help="where to write CSV + PNG")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lams = np.linspace(args.lam_min, args.lam_max, args.steps)

    csv_path = out_dir / "exact-qpt-curves.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_qubits", "h", "gamma", "lam", "E0", "E0_per_spin"])
        for n in args.n:
            for lam, e0 in zip(lams, exact_qpt_curve(n, args.h, lams, args.gamma), strict=True):
                writer.writerow(
                    [n, args.h, args.gamma, f"{lam:.6g}", f"{e0:.12g}", f"{e0 / n:.12g}"]
                )
    pdf_path = out_dir / "exact-qpt-curves.pdf"
    plot_exact_curves(args.n, args.h, lams, args.gamma, pdf_path)
    print(f"wrote {csv_path} and {pdf_path}")


def load_config_check(data: dict) -> None:
    """Validate that the non-sweep part of a sweep TOML is a well-formed RunConfig."""
    config_from_dict(data)


def sweep_plan_main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Expand a sweep TOML into a manifest CSV (one row per run)."
    )
    parser.add_argument("--config", required=True, help="sweep TOML: run config + [sweep] table")
    parser.add_argument("--runs-root", required=True, help="root directory for run outputs")
    parser.add_argument("--manifest", help="manifest path (default <runs-root>/manifest.csv)")
    args = parser.parse_args(argv)

    with open(args.config, "rb") as f:
        data = tomllib.load(f)
    sweep = data.pop("sweep", {})
    seeds = sweep.get("seeds", [1])
    grid: dict[str, list] = sweep.get("grid", {})
    variants: list[dict] = sweep.get("variants", [{"name": "default"}])

    load_config_check(data)
    Path(args.runs_root).mkdir(parents=True, exist_ok=True)

    grid_keys = list(grid)
    override_keys = sorted(
        set(grid_keys) | {k for v in variants for k in v if k != "name"} | {"train.seed"}
    )

    manifest_path = Path(args.manifest or Path(args.runs_root) / "manifest.csv")
    rows = []
    row_idx = 0
    for variant in variants:
        variant_overrides = {k: v for k, v in variant.items() if k != "name"}
        for combo in itertools.product(*(grid[k] for k in grid_keys)):
            grid_overrides = dict(zip(grid_keys, combo, strict=True))
            for seed in seeds:
                overrides = {**grid_overrides, **variant_overrides, "train.seed": seed}
                run_id = "_".join(
                    [variant["name"]]
                    + [f"{k.split('.')[-1]}{v}" for k, v in grid_overrides.items()]
                    + [f"s{seed}"]
                )
                rows.append(
                    {
                        "row": row_idx,
                        "run_id": run_id,
                        "config": str(Path(args.config).resolve()),
                        "out_dir": str(Path(args.runs_root) / run_id),
                        **{k: toml_literal(overrides.get(k)) for k in override_keys},
                    }
                )
                row_idx += 1

    seen_variants = set()
    for row in rows:
        variant = row["run_id"].split("_n_qubits")[0]
        if variant in seen_variants:
            continue
        seen_variants.add(variant)
        load_config(row["config"], manifest_row_to_overrides(row))

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(MANIFEST_FIXED_COLUMNS) + override_keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {manifest_path} with {len(rows)} runs (array range 0-{len(rows) - 1})")


def state_quality_main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="State-quality metrics (fidelity, order parameter, entropy) for a sweep."
    )
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--out", help="output CSV (default <sweep-dir>/state-results.csv)")
    args = parser.parse_args(argv)

    sweep_dir = Path(args.sweep_dir)
    df = pd.DataFrame(analyze_sweep(sweep_dir))
    results = sweep_dir / "results.csv"
    if results.exists():
        df = pd.read_csv(results).merge(df, on="run", how="left")
    out = Path(args.out or sweep_dir / "state-results.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} runs)")


def aggregate_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate a sweep into results.csv + figures.")
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--out", help="results CSV path (default <sweep-dir>/results.csv)")
    parser.add_argument("--plots", action="store_true", help="also write QPT error figure")
    args = parser.parse_args(argv)

    sweep_dir = Path(args.sweep_dir)
    df, incomplete = aggregate_sweep(sweep_dir)
    out = Path(args.out or sweep_dir / "results.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} complete runs)")
    if incomplete:
        print(f"WARNING: {len(incomplete)} incomplete runs: {', '.join(incomplete[:20])}")
    if args.plots and not df.empty:
        fig_path = sweep_dir / "qpt-errors.png"
        plot_qpt_errors(df, fig_path)
        print(f"wrote {fig_path}")
