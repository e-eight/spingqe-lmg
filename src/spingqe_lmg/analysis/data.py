"""Shared data-loading utilities for plot scripts and analysis pipelines."""

from pathlib import Path

import pandas as pd

from spingqe_lmg.analysis.aggregate import aggregate_sweep
from spingqe_lmg.analysis.plots import variant_column


def load_sweep_df(sweep_dir: str | Path) -> pd.DataFrame:
    """Aggregate a sweep directory and add ``variant`` and ``lam_over_h`` columns."""
    df, incomplete = aggregate_sweep(Path(sweep_dir))
    if incomplete:
        print(f"incomplete runs skipped in {sweep_dir}: {', '.join(incomplete[:20])}")
    df["variant"] = variant_column(df)
    df["lam_over_h"] = df["lam"] / df["h"]
    return df


def load_state_df(sweep_dir: str | Path) -> pd.DataFrame:
    """Load ``state-results.csv`` from a sweep directory and add derived columns."""
    path = Path(sweep_dir) / "state-results.csv"
    df = pd.read_csv(path)
    df["variant"] = variant_column(df)
    df["lam_over_h"] = df["lam"] / df["h"]
    if "seed" not in df.columns:
        df["seed"] = df["run"].str.extract(r"_s(\d+)$").astype(int)
    return df


def load_state_dfs(sweep_dirs: list[str | Path]) -> pd.DataFrame:
    """Concatenate ``state-results.csv`` from multiple sweep directories."""
    return pd.concat(
        [load_state_df(d) for d in sweep_dirs],
        ignore_index=True,
    )


def load_csv_df(path: str | Path) -> pd.DataFrame:
    """Load a CSV, add ``variant`` and ``lam_over_h`` columns.

    If the CSV already has a ``variant`` column (pre-aggregated data),
    it is preserved rather than recomputed.
    """
    df = pd.read_csv(Path(path))
    if "variant" not in df.columns:
        df["variant"] = variant_column(df)
    if "lam_over_h" not in df.columns:
        df["lam_over_h"] = df["lam"] / df["h"]
    return df
