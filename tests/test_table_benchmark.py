"""Tests for scripts/table-benchmark.py."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.script

CANONICAL = Path("data/tables/bench-merged-2026-07-08.csv")
SCALE_LQ = Path("data/tables/scale-lq-2026-07-08.csv")
SCALE_INCR = Path("data/tables/scale-incr-2026-07-08.csv")

needs_canonical = pytest.mark.skipif(
    not CANONICAL.exists(), reason="canonical benchmark CSV not present"
)


def _load_module():
    path = Path(__file__).parent.parent / "scripts" / "table-benchmark.py"
    spec = importlib.util.spec_from_file_location("table_benchmark", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["table_benchmark"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def tb():
    return _load_module()


def _throughput_csv(tmp_path, tb):
    rows = []
    for backend, n, val in [
        ("lightning.qubit (grouped)", 12, 0.082), ("lightning.qubit (grouped)", 24, None),
        ("incremental", 12, 0.0057), ("incremental", 24, 140.6),
        ("dicke", 12, 9.0e-5), ("dicke", 1000, 1.68e-2),
    ]:
        if val is None:
            continue
        rows.append(
            {
                "benchmark": "dicke" if backend == "dicke" else "backend",
                "n_qubits": n,
                "backend": backend,
                "seq_len": 12,
                "n_jobs": 1,
                "s_per_seq": val,
                "process_rss_gb": 1.0,
                "n_sequences": 20,
                "rep": 0,
            }
        )
    path = tmp_path / "bench.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_throughput_table_marks_known_oom_cell(tmp_path, tb):
    path = _throughput_csv(tmp_path, tb)
    df = tb.throughput_table(path)
    nseq_df = tb.n_seq_table(path)
    assert df.loc[24, "l.qubit (g)"] != df.loc[24, "l.qubit (g)"]  # NaN
    latex = tb.to_latex_throughput(df, nseq_df)
    assert r"\text{OOM}" in latex


def test_throughput_table_not_applicable_cell_is_dash(tmp_path, tb):
    path = _throughput_csv(tmp_path, tb)
    df = tb.throughput_table(path)
    nseq_df = tb.n_seq_table(path)
    latex = tb.to_latex_throughput(df, nseq_df)
    lines = latex.splitlines()
    n1000_line = next(line for line in lines if line.strip().startswith("1000"))
    assert "---" in n1000_line
    assert r"\text{OOM}" not in n1000_line


def test_throughput_table_marks_reduced_batch_size_with_dagger(tmp_path, tb):
    """CPU evaluators drop n_sequences from 20 to 1 at high N to bound
    subprocess wall-clock (documented in the 2026-07-08 canonical note's
    Methodology section) -- the LaTeX output must flag this, not silently
    mix batch sizes within a column."""
    rows = [
        {"benchmark": "backend", "n_qubits": 12, "backend": "incremental",
         "seq_len": 12, "n_jobs": 1, "s_per_seq": 0.0057, "n_sequences": 20, "rep": 0},
        {"benchmark": "backend", "n_qubits": 24, "backend": "incremental",
         "seq_len": 12, "n_jobs": 1, "s_per_seq": 140.6, "n_sequences": 1, "rep": 0},
    ]
    path = tmp_path / "bench.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    df = tb.throughput_table(path)
    nseq_df = tb.n_seq_table(path)
    latex = tb.to_latex_throughput(df, nseq_df)

    lines = latex.splitlines()
    n12_line = next(line for line in lines if line.strip().startswith("12"))
    n24_line = next(line for line in lines if line.strip().startswith("24"))
    assert r"^\dagger" not in n12_line, "n_seq=20 (modal) cell must not be marked"
    assert r"^\dagger" in n24_line, "n_seq=1 (reduced-batch) cell must be marked"


def test_scaling_summary_reports_peak_and_floor(tmp_path, tb):
    l_rows = [
        {"benchmark": "scaling", "n_jobs": 1, "wall_clock_s": 100.0, "rep": 0},
        {"benchmark": "scaling", "n_jobs": 8, "wall_clock_s": 20.0, "rep": 0},
        {"benchmark": "scaling", "n_jobs": 32, "wall_clock_s": 30.0, "rep": 0},
    ]
    s_rows = [
        {"benchmark": "scaling", "n_jobs": 1, "wall_clock_s": 100.0, "rep": 0},
        {"benchmark": "scaling", "n_jobs": 8, "wall_clock_s": 10.0, "rep": 0},
        {"benchmark": "scaling", "n_jobs": 32, "wall_clock_s": 12.0, "rep": 0},
    ]
    l_path = tmp_path / "l.csv"
    s_path = tmp_path / "s.csv"
    pd.DataFrame(l_rows).to_csv(l_path, index=False)
    pd.DataFrame(s_rows).to_csv(s_path, index=False)

    summary = tb.scaling_summary(l_path, s_path)

    assert summary["lightning_peak_speedup"] == pytest.approx(5.0)
    assert summary["lightning_peak_n_jobs"] == 8
    assert summary["incremental_peak_speedup"] == pytest.approx(10.0)
    assert summary["incremental_peak_n_jobs"] == 8
    assert summary["floor_ratio"] == pytest.approx(2.0)


@needs_canonical
def test_bench_macros_known_values(tb):
    out = tb.bench_macros(CANONICAL, SCALE_LQ, SCALE_INCR)
    # 0.082106 / 0.005658 = 14.51 -> 15 at 2 sig figs (full-precision CSV
    # medians; the plan's "~14" note used pre-rounded 8.21e-2/5.7e-3).
    assert r"\newcommand{\bmkIncrOverLqDefaultAtTwelve}{15}" in out
    assert r"\newcommand{\bmkGpuCrossoverN}{14}" in out
    # Last CPU-fastest N: the measured grid point below the GPU crossover.
    assert r"\newcommand{\bmkCpuRegimeMaxN}{12}" in out
    assert r"\newcommand{\bmkLgpuCrossoverN}{24}" in out
    assert r"\newcommand{\bmkPevenGpuOverCpuAtMax}{1000}" in out
    assert r"\newcommand{\bmkOursOverLgpuGroupedMax}{6.3}" in out
    assert r"\newcommand{\bmkScalingPeakIncr}{9.2}" in out
    assert out.startswith("% AUTO-GENERATED")
    # Honest workload (2026-07-12 review): 700 epochs x 10 candidates
    # + 14 x 100-sequence eval passes = 8400 sequences per run.
    # threshold = 2.0 h * 3600 * 16 workers / 8400 = 13.71 -> "14".
    assert r"\newcommand{\bmkTractableSeqSec}{14}" in out
    assert r"\newcommand{\bmkWallCpuIncr}{20}" in out
    assert r"\newcommand{\bmkWallCpuPeven}{22}" in out
    assert r"\newcommand{\bmkWallCpuLqGrouped}{22}" in out
    assert r"\newcommand{\bmkWallGpu}{26}" in out
    assert "bmkWallCpuCustom" not in out


def test_sig2_formatting(tb):
    assert tb._sig2(1046.2) == "1000"
    assert tb._sig2(726.7) == "730"
    assert tb._sig2(14.41) == "14"
    assert tb._sig2(2.33) == "2.3"
    assert tb._sig2(6.27) == "6.3"
