#!/usr/bin/env python
r"""Generate LaTeX table rows for the evaluator throughput and peak-RSS tables,
and strong-scaling summary numbers, from the canonical benchmark CSVs.

Every value is a median over the `rep` repetitions in the canonical
2026-07-08 benchmark CSVs --- never hand-typed. The paper's throughput table
(tab:throughput) prints four of the `throughput` columns: l.gpu (g),
Incr GPU, P.-E. GPU and Symm.

Usage::

    python scripts/table-benchmark.py throughput \
        data/tables/bench-merged-2026-07-08.csv --latex

    python scripts/table-benchmark.py rss \
        data/tables/bench-merged-2026-07-08.csv --latex

    python scripts/table-benchmark.py scaling-summary \
        data/tables/scale-lq-2026-07-08.csv \
        data/tables/scale-incr-2026-07-08.csv

    python scripts/table-benchmark.py macros \
        data/tables/bench-merged-2026-07-08.csv \
        data/tables/scale-lq-2026-07-08.csv \
        data/tables/scale-incr-2026-07-08.csv

The `macros` subcommand writes `bench-macros.tex`, a set of `\newcommand`
ratios (2 significant figures) for quoting benchmark ratios in prose.
"""

from __future__ import annotations

import argparse
import sys
from math import floor, log10
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from spingqe_lmg.analysis import load_backend_series, load_scaling_series  # noqa: E402

_THROUGHPUT_COLUMNS = [
    ("lightning.qubit (grouped)", "l.qubit (g)"),
    ("incremental", "Incr. CPU"),
    ("parity_even", "P.-E. CPU"),
    ("lightning.gpu (grouped)", "l.gpu (g)"),
    ("incremental_gpu", "Incr. GPU"),
    ("parity_even_gpu", "P.-E. GPU"),
    ("dicke", "Symm. sector"),
]
_THROUGHPUT_NS = [12, 14, 16, 18, 20, 22, 24, 26, 1000]

_RSS_COLUMNS = [
    ("lightning.qubit (grouped)", "l.qubit (g)"),
    ("incremental", "Incr. CPU"),
    ("parity_even", "P.-E. CPU"),
    ("dicke", "Symm. sector"),
]
_RSS_NS = [12, 20, 24, 26, 1000]

_KNOWN_OOM = {
    ("lightning.qubit (grouped)", 24),
    ("lightning.qubit (grouped)", 26),
}


def _pivot(
    csv_path: str | Path, columns: list[tuple[str, str]], ns: list[int], value_col: str
) -> pd.DataFrame:
    codes = [code for code, _ in columns]
    series = load_backend_series(
        csv_path, seq_len=12, max_n=None, include_symm="dicke" in codes,
        skip=set(),
    )
    by_code = {code: dict(zip(n_list, y_list, strict=True)) for code, n_list, y_list in series}
    data = {
        label: [by_code.get(code, {}).get(n, float("nan")) for n in ns]
        for code, label in columns
    }
    return pd.DataFrame(data, index=ns)


def throughput_table(csv_path: str | Path) -> pd.DataFrame:
    return _pivot(csv_path, _THROUGHPUT_COLUMNS, _THROUGHPUT_NS, "s_per_seq")


_MODAL_N_SEQ = 20


def n_seq_table(csv_path: str | Path) -> pd.DataFrame:
    """Return the `n_sequences` batch size used for each throughput cell."""
    df = pd.read_csv(csv_path)
    df = df[df["seq_len"] == 12]
    data = {}
    for code, label in _THROUGHPUT_COLUMNS:
        benchmark = "dicke" if code == "dicke" else "backend"
        sub = df[(df["benchmark"] == benchmark) & (df["backend"] == code)]
        nseq = sub.groupby("n_qubits")["n_sequences"].first()
        data[label] = [
            int(nseq[n]) if n in nseq.index else None for n in _THROUGHPUT_NS
        ]
    return pd.DataFrame(data, index=_THROUGHPUT_NS)


def rss_table(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["seq_len"] == 12]
    data = {}
    for code, label in _RSS_COLUMNS:
        benchmark = "dicke" if code == "dicke" else "backend"
        sub = df[(df["benchmark"] == benchmark) & (df["backend"] == code)]
        med = sub.groupby("n_qubits")["process_rss_gb"].median()
        data[label] = [float(med.get(n, float("nan"))) for n in _RSS_NS]
    return pd.DataFrame(data, index=_RSS_NS)


def _fmt_cell(value: float, backend_code: str, n: int) -> str:
    if value != value:  # NaN
        if (backend_code, n) in _KNOWN_OOM:
            return r"\text{OOM}"
        return "---"
    if value >= 100:
        return rf"\( \num{{{value:.1e}}} \)"
    return rf"\( \num{{{value:.2e}}} \)" if value < 1 else rf"\( \num{{{value:.1f}}} \)"


def to_latex_throughput(df: pd.DataFrame, nseq_df: pd.DataFrame) -> str:
    label_to_code = {label: code for code, label in _THROUGHPUT_COLUMNS}
    lines = []
    for n in df.index:
        cells = []
        for label in df.columns:
            cell = _fmt_cell(df.loc[n, label], label_to_code[label], n)
            nseq = nseq_df.loc[n, label]
            if (
                cell not in ("---", r"\text{OOM}")
                and nseq is not None
                and int(nseq) != _MODAL_N_SEQ
            ):
                cell = cell[:-3] + r"{}^\dagger \)"
            cells.append(cell)
        lines.append(f"        {n:<5}" + " & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def to_latex_rss(df: pd.DataFrame) -> str:
    label_to_code = {label: code for code, label in _RSS_COLUMNS}
    lines = []
    for n in df.index:
        cells = [_fmt_cell(df.loc[n, label], label_to_code[label], n) for label in df.columns]
        lines.append(f"        {n:<5}" + " & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def scaling_summary(lightning_csv: str | Path, incremental_csv: str | Path) -> dict:
    n_jobs, lightning_s, incremental_s = load_scaling_series(lightning_csv, incremental_csv)
    l0, s0 = lightning_s[0], incremental_s[0]
    l_speedup = [l0 / v for v in lightning_s]
    s_speedup = [s0 / v for v in incremental_s]
    l_peak_i = max(range(len(l_speedup)), key=lambda i: l_speedup[i])
    s_peak_i = max(range(len(s_speedup)), key=lambda i: s_speedup[i])
    l_floor = min(lightning_s)
    s_floor = min(incremental_s)
    return {
        "n_jobs": n_jobs,
        "lightning_peak_speedup": l_speedup[l_peak_i],
        "lightning_peak_n_jobs": n_jobs[l_peak_i],
        "incremental_peak_speedup": s_speedup[s_peak_i],
        "incremental_peak_n_jobs": n_jobs[s_peak_i],
        "lightning_floor_s": l_floor,
        "incremental_floor_s": s_floor,
        "floor_ratio": l_floor / s_floor,
    }


def _sig2(x: float) -> str:
    """Format to 2 significant figures, no scientific notation (ratios only)."""
    digits = 1 - int(floor(log10(abs(x))))
    return f"{round(x, digits):g}"


# Actual training workload (uniform across all runs' config.json):
# 10 candidates per epoch, plus a 100-sequence diagnostic evaluation
# every 50 epochs. 700*10 + (700//50)*100 = 8400 sequences per run.
_EPOCHS, _SEQ_PER_EPOCH, _EVAL_EVERY, _EVAL_SEQ = 700, 10, 50, 100
_WORKERS, _BUDGET_H = 16, 2.0
_TOTAL_SEQ = _EPOCHS * _SEQ_PER_EPOCH + (_EPOCHS // _EVAL_EVERY) * _EVAL_SEQ


def _median_map(csv_path: str | Path) -> dict[str, dict[int, float]]:
    series = load_backend_series(
        csv_path, seq_len=12, max_n=None, include_symm=True, skip=set()
    )
    return {code: dict(zip(n_list, y_list, strict=True)) for code, n_list, y_list in series}


def bench_macros(
    bench_csv: str | Path, lightning_csv: str | Path, incremental_csv: str | Path
) -> str:
    m = _median_map(bench_csv)

    def r(num: str, den: str, n: int) -> float:
        return m[num][n] / m[den][n]

    gpu_cross = min(
        n
        for n in sorted(m["parity_even_gpu"])
        if n in m["parity_even"]
        and n in m["incremental"]
        and m["parity_even_gpu"][n] < m["parity_even"][n]
        and m["incremental_gpu"][n] < m["incremental"][n]
    )
    # Largest measured N where a custom CPU evaluator is still fastest
    # (the grid point below the GPU crossover). Prose/tables must use this
    # for "CPU regime" upper bounds; gpu_cross is the FIRST GPU-winning N.
    cpu_regime_max = max(n for n in m["parity_even"] if n < gpu_cross)
    lgpu_cross = min(
        n
        for n in sorted(m["lightning.gpu (grouped)"])
        if n in m["parity_even_gpu"]
        and n in m["incremental_gpu"]
        and m["lightning.gpu (grouped)"][n] < m["parity_even_gpu"][n]
        and m["lightning.gpu (grouped)"][n] < m["incremental_gpu"][n]
    )
    ours_max = max(
        m["lightning.gpu (grouped)"][n] / m[ours][n]
        for ours in ("incremental_gpu", "parity_even_gpu")
        for n in m[ours]
        if 14 <= n <= 22 and n in m["lightning.gpu (grouped)"]
    )
    lgpu_grouping_gain_max = max(
        m["lightning.gpu"][n] / m["lightning.gpu (grouped)"][n]
        for n in m["lightning.gpu"]
        if n in m["lightning.gpu (grouped)"]
    )
    threshold = (_BUDGET_H * 3600 * _WORKERS) / _TOTAL_SEQ

    def wall(codes: tuple[str, ...]) -> int:
        return max(
            n
            for n in m[codes[0]]
            if n <= 26 and all(n in m[c] and m[c][n] <= threshold for c in codes)
        )

    scale = scaling_summary(lightning_csv, incremental_csv)
    cudaq = [v for n, v in m["cudaq_v1"].items() if 12 <= n <= 26]

    macros: dict[str, float | int] = {
        "bmkIncrOverLqDefaultAtTwelve": r("lightning.qubit", "incremental", 12),
        "bmkIncrOverLqGroupedAtTwelve": r("lightning.qubit (grouped)", "incremental", 12),
        "bmkPevenOverLqGroupedAtTwelve": r("lightning.qubit (grouped)", "parity_even", 12),
        "bmkGpuCrossoverN": gpu_cross,
        "bmkCpuRegimeMaxN": cpu_regime_max,
        "bmkPevenGpuOverCpuAtCross": r("parity_even", "parity_even_gpu", gpu_cross),
        "bmkPevenGpuOverCpuAtMax": r("parity_even", "parity_even_gpu", 26),
        "bmkIncrGpuOverCpuAtMax": r("incremental", "incremental_gpu", 24),
        "bmkOursOverLgpuGroupedMax": ours_max,
        "bmkLgpuCrossoverN": lgpu_cross,
        "bmkLgpuGroupedOverPevenGpuAtTwentySix": r(
            "parity_even_gpu", "lightning.gpu (grouped)", 26
        ),
        "bmkLgpuGroupedOverIncrGpuAtTwentySix": r(
            "incremental_gpu", "lightning.gpu (grouped)", 26
        ),
        "bmkSymmOverBestFullHilbertAtTwelve": r("parity_even", "dicke", 12),
        "bmkSymmOverIncrAtTwelve": r("incremental", "dicke", 12),
        "bmkCudaqFloorMin": min(cudaq),
        "bmkCudaqFloorMax": max(cudaq),
        "bmkScalingPeakIncr": scale["incremental_peak_speedup"],
        "bmkScalingPeakIncrJobs": scale["incremental_peak_n_jobs"],
        "bmkScalingPeakLq": scale["lightning_peak_speedup"],
        "bmkScalingPeakLqJobs": scale["lightning_peak_n_jobs"],
        "bmkScalingFloorRatio": scale["floor_ratio"],
        "bmkTractableSeqSec": threshold,
        "bmkWallCpuIncr": wall(("incremental",)),
        "bmkWallCpuPeven": wall(("parity_even",)),
        "bmkWallCpuLqGrouped": wall(("lightning.qubit (grouped)",)),
        "bmkWallGpu": wall(
            ("incremental_gpu", "parity_even_gpu", "lightning.gpu (grouped)")
        ),
        "bmkLgpuGroupingGainMax": lgpu_grouping_gain_max,
    }
    lines = [
        "% AUTO-GENERATED by `python scripts/table-benchmark.py macros` -- DO NOT EDIT.",
        "% Source: " + str(bench_csv),
    ]
    for name, value in macros.items():
        text = str(value) if isinstance(value, int) else _sig2(float(value))
        lines.append(rf"\newcommand{{\{name}}}{{{text}}}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_thr = sub.add_parser("throughput")
    p_thr.add_argument("csv")
    p_thr.add_argument("--latex", action="store_true")

    p_rss = sub.add_parser("rss")
    p_rss.add_argument("csv")
    p_rss.add_argument("--latex", action="store_true")

    p_scale = sub.add_parser("scaling-summary")
    p_scale.add_argument("lightning_csv")
    p_scale.add_argument("incremental_csv")

    p_mac = sub.add_parser("macros")
    p_mac.add_argument("bench_csv")
    p_mac.add_argument("lightning_csv")
    p_mac.add_argument("incremental_csv")
    p_mac.add_argument("-o", "--out", default="bench-macros.tex")

    args = ap.parse_args(argv)

    if args.cmd == "throughput":
        df = throughput_table(args.csv)
        if not args.latex:
            print(df)
        else:
            print(to_latex_throughput(df, n_seq_table(args.csv)))
    elif args.cmd == "rss":
        df = rss_table(args.csv)
        print(df if not args.latex else to_latex_rss(df))
    elif args.cmd == "scaling-summary":
        summary = scaling_summary(args.lightning_csv, args.incremental_csv)
        for key, value in summary.items():
            print(f"{key}: {value}")
    elif args.cmd == "macros":
        text = bench_macros(args.bench_csv, args.lightning_csv, args.incremental_csv)
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
