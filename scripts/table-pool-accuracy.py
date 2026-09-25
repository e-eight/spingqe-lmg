#!/usr/bin/env python
r"""Generate LaTeX table rows for the pool-variant accuracy table (Table VI in the paper).

Reads one or more ``results.csv`` files (or ``combined-results.csv``) from GQE
sweeps, groups by (N, lambda, pool variant), computes the best-over-seeds
refined relative error, and prints formatted LaTeX rows for the target
combinations.

Usage::

    # The paper's collective-pool accuracy table (Table VI)
    python scripts/table-pool-accuracy.py \
        data/tables/pool-accuracy-l12-s1-10-2026-07-29.csv \
        --latex --collective-only --floor 1e-10 --stat median

Supports ``--pivot`` for a human-readable pivot table, ``--pool-size``
for pool-size statistics, and ``--seed-counts`` for seed-coverage audit.
The ``--latex`` output is the ``\midrule ... \bottomrule`` table body.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Pool-variant detection
# ---------------------------------------------------------------------------
# When reading results.csv files the variant column may be absent; when reading
# combined-results.csv it is already present.  This module detects variants
# from the `run` column when needed, preferring the canonical `variant` column
# when it exists.
# ---------------------------------------------------------------------------

_VARIANT_ORDER = ["collective", "pairwise-ext-mf", "pairwise-all"]

# Prose labels used in the LaTeX table header.
_VARIANT_LABEL = {
    "collective": "collective",
    "pairwise-ext-mf": "ext.\\ pairwise",
    "pairwise-all": "std.\\ pairwise",
}


def _variant_from_run(df: pd.DataFrame) -> pd.Series:
    """Assign pool variant based on the run-name convention.

    Returns a ``pd.Series`` with values ``"collective"``,
    ``"pairwise-ext-mf"``, or ``"pairwise-all"``.  Nearest-neighbour runs
    (``pairwise-nn_*``) are mapped to ``None`` so they can be dropped.
    """
    run = df["run"].astype(str)
    return pd.Series(
        pd.Categorical(
            run.map(_classify_run),
            categories=_VARIANT_ORDER,
            ordered=True,
        ),
        index=df.index,
        name="variant",
    )


def _classify_run(run_name: str) -> str | None:
    if run_name.startswith("collective"):
        return "collective"
    if "ext-mf" in run_name:
        return "pairwise-ext-mf"
    if "pairwise-nn" in run_name:
        return None  # nearest-neighbour — drop
    if run_name.startswith("pairwise"):
        return "pairwise-all"
    return None


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def load_csvs(paths: list[str | Path], seq_len: int | None = None) -> pd.DataFrame:
    """Load one or more ``results.csv`` (or ``combined-results.csv``) files.

    Prefers the canonical ``variant`` column when present; falls back to
    run-name heuristics.  Filters to *seq_len* when provided.

    Accepts both the sweep-native column names (``variant``, ``n_qubits``,
    ``lam``) and the canonical frozen-CSV names (``pool``, ``N``, ``lambda``),
    and skips the provenance comment line those frozen files carry.
    """
    frames: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(Path(p), comment="#")
        df = df.rename(columns={"pool": "variant", "N": "n_qubits", "lambda": "lam"})
        if "variant" not in df.columns:
            df["variant"] = _variant_from_run(df)
        df = df.dropna(subset=["variant"])
        if "seq_len" in df.columns and seq_len is not None:
            df = df[df["seq_len"] == seq_len]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def best_over_seeds(
    df: pd.DataFrame, stat: str = "best", floor: float = 1e-10
) -> pd.DataFrame:
    """Group by (N, lam, variant) and reduce the per-seed errors over seeds.

    *stat* is ``"best"`` (min, the historical default) or ``"median"``
    (the seed-robust statistic).  Both the refined error and the
    raw pre-refinement error are reduced with the same statistic, and the
    across-seed min and max of the refined error are carried alongside so a
    caller can report a spread with the central value.

    *floor* is the convergence floor: ``floor_hits`` counts the seeds in each
    cell at or below it, which is what marks a cell with a dagger in the
    manuscript's collective-pool table.
    """
    agg = {"best": "min", "median": "median"}[stat]
    group_cols = ["n_qubits", "lam", "variant"]
    aggs = {
        "refined_rel_error": ("refined_rel_error", agg),
        "refined_rel_error_lo": ("refined_rel_error", "min"),
        "refined_rel_error_hi": ("refined_rel_error", "max"),
        "n_seeds": ("seed", "nunique"),
        "floor_hits": ("refined_rel_error", lambda s: int((s <= floor).sum())),
    }
    # Not every caller carries these: the frozen canonical CSVs have no
    # pool_size, and frames built from refined errors alone have no rel_error.
    if "rel_error" in df.columns:
        aggs["raw_rel_error"] = ("rel_error", agg)
    if "pool_size" in df.columns:
        aggs["pool_size"] = ("pool_size", "first")
    return df.groupby(group_cols, observed=False).agg(**aggs).reset_index()


def _fmt(x: float | None) -> str:
    """Format a relative error value for LaTeX.

    No low-end clipping: the collective pool's best-over-seeds error reaches
    \\(3\\times10^{-13}\\) at \\(N=8\\), and collapsing everything below
    \\(10^{-8}\\) to one bucket would erase the range the table reports.
    """
    if x is None or pd.isna(x):
        return "---"
    return rf"\( \num{{{x:.1e}}} \)"


def _fmt_bare(x: float | None) -> str:
    """Format a value for use inside an enclosing math-mode range."""
    if x is None or pd.isna(x):
        return "---"
    if x < 1e-8:
        return r"<\num{1e-8}"
    return rf"\num{{{x:.1e}}}"


def _fmt_range(lo: float | None, hi: float | None) -> str:
    """Format an across-seed [min, max] range for LaTeX."""
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return "---"
    return rf"\( [{_fmt_bare(lo)}, {_fmt_bare(hi)}] \)"


def _fmt_spread(lo: float | None, hi: float | None) -> str:
    """Format the across-seed spread of the refined error as a ratio."""
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return "---"
    # Both seeds at the float64 floor: the spread is not resolvable.
    if hi < 1e-8:
        return r"\( - \)"
    if lo <= 0 or lo < 1e-16:
        return r"\( > \num{1e8} \)"
    return rf"\( \num{{{hi / lo:.0f}}} \)"


def to_latex(
    df: pd.DataFrame,
    targets: list[tuple[int, float]] | None = None,
    with_raw: bool = False,
    collective_only: bool = False,
) -> str:
    r"""Format a seed-reduced DataFrame as LaTeX table body rows.

    Parameters
    ----------
    df : DataFrame
        Output of ``best_over_seeds()``.
    targets : list of (N, lam) tuples, optional
        If provided, emit exactly these rows in this order.  Missing
        combinations produce ``---`` columns.
    with_raw : bool
        When true, emit three cells per pool: the raw pre-refinement error, the
        refined error, and the across-seed spread of the refined error.  When
        false, emit only the refined error, which is the historical layout.
    collective_only : bool
        When true, emit the four-column collective-pool layout used by
        Table VI: N, lambda, the stat-selected central value, and
        the best over seeds, the last daggered where any seed reached the
        convergence floor.  Takes precedence over *with_raw*.
    """
    if targets is None:
        targets = sorted(
            df[["n_qubits", "lam"]].drop_duplicates().itertuples(index=False, name=None)
        )

    values = ["refined_rel_error"]
    if collective_only:
        values += ["refined_rel_error_lo", "floor_hits"]
    elif with_raw:
        values += ["raw_rel_error", "refined_rel_error_lo", "refined_rel_error_hi"]
    pivot = df.pivot_table(
        index=["n_qubits", "lam"],
        columns="variant",
        values=values,
        aggfunc="first",
    )

    def cell(value: str, variant: str, n: int, lam: float) -> float | None:
        key = (value, variant)
        if key not in pivot.columns or (n, lam) not in pivot.index:
            return None
        return pivot.loc[(n, lam), key]

    if collective_only:
        n_cols = 4
    else:
        n_cols = 2 + len(_VARIANT_ORDER) * (3 if with_raw else 1)

    lines: list[str] = []
    prev_n: int | None = None
    for n, lam in targets:
        # Suppress N on rows after the first in each group.
        cells = [str(n) if n != prev_n else "", str(lam)]
        if collective_only:
            cells.append(_fmt(cell("refined_rel_error", "collective", n, lam)))
            best = _fmt(cell("refined_rel_error_lo", "collective", n, lam))
            hits = cell("floor_hits", "collective", n, lam)
            if hits is not None and not pd.isna(hits) and hits > 0:
                best += r"\(^\dagger\)"
            cells.append(best)
            lines.append("        " + " & ".join(cells) + r" \\")
            prev_n = n
            continue
        for v in _VARIANT_ORDER:
            if with_raw:
                cells.append(_fmt(cell("raw_rel_error", v, n, lam)))
            cells.append(_fmt(cell("refined_rel_error", v, n, lam)))
            if with_raw:
                cells.append(
                    _fmt_spread(
                        cell("refined_rel_error_lo", v, n, lam),
                        cell("refined_rel_error_hi", v, n, lam),
                    )
                )
        lines.append("        " + " & ".join(cells) + r" \\")
        prev_n = n

    # Insert a \cmidrule between N groups.
    result_lines: list[str] = []
    for i, line in enumerate(lines):
        result_lines.append(line)
        if i + 1 < len(lines) and targets[i + 1][0] != targets[i][0]:
            result_lines.append(rf"        \cmidrule{{2-{n_cols}}}")
    return "\n".join(result_lines)


def to_pivot(df: pd.DataFrame) -> str:
    """Human-readable pivot table printed to stderr for inspection."""
    pivot = df.pivot_table(
        index=["n_qubits", "lam"],
        columns="variant",
        values="refined_rel_error",
        aggfunc="first",
    )
    pivot = pivot.reindex(columns=_VARIANT_ORDER)
    with pd.option_context("display.float_format", "{:.2e}".format, "display.max_rows", None):
        return str(pivot)


def to_pool_sizes(df: pd.DataFrame) -> str:
    """Show pool sizes per (N, variant) for context."""
    ps = df.groupby(["n_qubits", "variant"], observed=False)["pool_size"].first().unstack()
    ps = ps.reindex(columns=_VARIANT_ORDER)
    return str(ps)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "csv", nargs="+", help="One or more results.csv files (or combined-results.csv)"
    )
    ap.add_argument("--latex", action="store_true", help="Output LaTeX table body rows")
    ap.add_argument("--pivot", action="store_true", help="Print human-readable pivot table")
    ap.add_argument("--pool-size", action="store_true", help="Show pool sizes per (N, variant)")
    ap.add_argument("--seed-counts", action="store_true", help="Show seed counts per cell")
    ap.add_argument(
        "--seq-len", type=int, default=12, help="Filter to this sequence length (default 12)"
    )
    ap.add_argument(
        "--extended",
        action="store_true",
        help="Include N>16 rows (18,20,22) for L=12; pairwise-all omitted where absent",
    )
    ap.add_argument(
        "--stat",
        choices=["best", "median"],
        default="best",
        help="Seed-reduction statistic: 'best' (min, historical default) or "
        "'median' (seed-robust qualifier for prose ordering claims)",
    )
    ap.add_argument(
        "--with-raw",
        action="store_true",
        help="Emit raw pre-refinement error and across-seed spread alongside "
        "the refined error (three cells per pool)",
    )
    ap.add_argument(
        "--collective-only",
        action="store_true",
        help="Emit the four-column collective-pool layout (N, lambda, central, "
        "best) with a dagger where any seed reached the convergence floor; "
        "also drops lambda=0.5, which exists only at N=8,10",
    )
    ap.add_argument(
        "--floor",
        type=float,
        default=1e-10,
        help="Convergence floor for the dagger count (default 1e-10, the "
        "reachability scanner's success threshold)",
    )
    args = ap.parse_args(argv)

    df = load_csvs(args.csv, seq_len=args.seq_len)
    best = best_over_seeds(df, stat=args.stat, floor=args.floor)

    if args.pool_size:
        print(to_pool_sizes(df))

    if args.seed_counts:
        sc = seed_count_table(df)
        print(sc)

    if args.pivot:
        print(to_pivot(best))

    if args.latex:
        lambdas = (1.0, 1.5, 2.0) if args.collective_only else (0.5, 1.0, 1.5, 2.0)
        targets = _build_targets(df, extended=args.extended, lambdas=lambdas)
        print(
            to_latex(
                best,
                targets=targets,
                with_raw=args.with_raw,
                collective_only=args.collective_only,
            )
        )


def _build_targets(
    df: pd.DataFrame,
    extended: bool = False,
    lambdas: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
) -> list[tuple[int, float]]:
    """Return ordered (N, lam) pairs for the pool-accuracy grid.

    Standard (extended=False): N=8,10,12,14,16 over *lambdas*.
    Extended: adds N=18,20,22 with the same λ grid.
    """
    targets: list[tuple[int, float]] = []
    n_values = [8, 10, 12, 14, 16]
    if extended:
        n_values += [18, 20, 22]
    for n in n_values:
        for lam in lambdas:
            targets.append((n, lam))
    return targets


def seed_count_table(df: pd.DataFrame) -> str:
    """Human-readable seed-count audit per (N, lam, variant)."""
    sc = (
        df.groupby(["n_qubits", "lam", "variant"], observed=False)["seed"]
        .nunique()
        .unstack(fill_value=0)
    )
    sc = sc.reindex(columns=_VARIANT_ORDER, fill_value=0)
    return str(sc)


if __name__ == "__main__":
    main()
