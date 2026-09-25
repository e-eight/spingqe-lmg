"""Shared CSV-loading and aggregation helpers for benchmark plots and tables.

Every canonical benchmark CSV (data/tables/bench-merged-*.csv,
data/tables/scale-*.csv) carries a
`rep` column: one row per independent repetition at a distinct seed. Every
loader here reduces across `rep` before returning a value (median by
default, matching the documented methodology), so a script never mistakes
repetition rows for distinct (N, backend) or (N, n_jobs) data points.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_AGG_FUNCS = {"median": "median", "mean": "mean"}


def _reduce(grouped: pd.core.groupby.SeriesGroupBy, agg: str) -> pd.Series:
    if agg not in _AGG_FUNCS:
        raise ValueError(f"Unknown agg {agg!r}; use 'median' or 'mean'")
    return getattr(grouped, _AGG_FUNCS[agg])()


def load_backend_series(
    csv_path: str | Path,
    seq_len: int | None = 12,
    max_n: int | None = None,
    include_symm: bool = True,
    skip: set[str] | None = None,
    agg: str = "median",
) -> list[tuple[str, list[int], list[float]]]:
    """Load per-N throughput series from a benchmark CSV.

    Returns one ``(backend_code, ns, ys)`` tuple per backend present in the
    ``"backend"`` rows and not in *skip*, plus one more for the
    ``"dicke"`` (permutation-symmetric sector) rows when *include_symm* is
    set. ``ys`` is ``s_per_seq``, reduced across the `rep` column via *agg*.
    """
    df = pd.read_csv(csv_path)
    if seq_len is not None and "seq_len" in df.columns:
        df = df[df["seq_len"] == seq_len]
    if "n_jobs" in df.columns:
        df = df[df["n_jobs"] == 1]

    skip = skip or set()
    series: list[tuple[str, list[int], list[float]]] = []

    df_b = df[df["benchmark"] == "backend"]
    for code in sorted(df_b["backend"].unique()):
        if code in skip:
            continue
        sub = _reduce(df_b[df_b["backend"] == code].groupby("n_qubits")["s_per_seq"], agg)
        if sub.empty:
            continue
        ns = sorted(int(n) for n in sub.index)
        if max_n is not None:
            ns = [n for n in ns if n <= max_n]
        ys = [float(sub[n]) for n in ns]
        if ns:
            series.append((code, ns, ys))

    if include_symm:
        df_d = df[df["benchmark"] == "dicke"]
        sub_d = _reduce(df_d.groupby("n_qubits")["s_per_seq"], agg)
        if not sub_d.empty:
            ns_d = sorted(int(n) for n in sub_d.index)
            if max_n is not None:
                ns_d = [n for n in ns_d if n <= max_n]
            ys_d = [float(sub_d[n]) for n in ns_d]
            if ns_d:
                series.append(("dicke", ns_d, ys_d))

    return series


def load_scaling_series(
    lightning_csv: str | Path,
    incremental_csv: str | Path,
    agg: str = "median",
) -> tuple[list[int], list[float], list[float]]:
    """Load strong-scaling wall-clock series, reduced across `rep`.

    Returns ``(n_jobs, lightning_wall_s, incremental_wall_s)``, one value
    per ``n_jobs`` present in both CSVs, reduced via *agg* (median by
    default).
    """
    df_l = pd.read_csv(lightning_csv)
    df_s = pd.read_csv(incremental_csv)

    sub_l = _reduce(df_l[df_l["benchmark"] == "scaling"].groupby("n_jobs")["wall_clock_s"], agg)
    sub_s = _reduce(df_s[df_s["benchmark"] == "scaling"].groupby("n_jobs")["wall_clock_s"], agg)

    n_jobs = sorted(int(n) for n in sub_l.index)
    if sorted(int(n) for n in sub_s.index) != n_jobs:
        raise ValueError(
            "lightning and incremental CSVs cover different n_jobs values: "
            f"{sorted(int(n) for n in sub_l.index)} vs "
            f"{sorted(int(n) for n in sub_s.index)}"
        )

    lightning_s = [float(sub_l[n]) for n in n_jobs]
    incremental_s = [float(sub_s[n]) for n in n_jobs]
    return n_jobs, lightning_s, incremental_s
