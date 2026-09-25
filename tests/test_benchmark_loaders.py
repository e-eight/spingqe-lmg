"""Tests for spingqe_lmg.analysis.benchmark_loaders."""

import pandas as pd
import pytest

from spingqe_lmg.analysis.benchmark_loaders import load_backend_series, load_scaling_series

pytestmark = pytest.mark.script


def _backend_csv(tmp_path, rows):
    path = tmp_path / "backend.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_backend_series_medians_over_reps(tmp_path):
    rows = [
        {"benchmark": "backend", "n_qubits": 12, "backend": "incremental",
         "seq_len": 12, "n_jobs": 1, "s_per_seq": 0.010, "rep": 0},
        {"benchmark": "backend", "n_qubits": 12, "backend": "incremental",
         "seq_len": 12, "n_jobs": 1, "s_per_seq": 0.012, "rep": 1},
        {"benchmark": "backend", "n_qubits": 12, "backend": "incremental",
         "seq_len": 12, "n_jobs": 1, "s_per_seq": 0.100, "rep": 2},
    ]
    path = _backend_csv(tmp_path, rows)
    series = load_backend_series(path, include_symm=False)
    assert series == [("incremental", [12], [0.012])]


def test_load_backend_series_respects_skip_and_max_n(tmp_path):
    rows = [
        {"benchmark": "backend", "n_qubits": 12, "backend": "cudaq_v1",
         "seq_len": 12, "n_jobs": 1, "s_per_seq": 0.2, "rep": 0},
        {"benchmark": "backend", "n_qubits": 12, "backend": "incremental",
         "seq_len": 12, "n_jobs": 1, "s_per_seq": 0.01, "rep": 0},
        {"benchmark": "backend", "n_qubits": 26, "backend": "incremental",
         "seq_len": 12, "n_jobs": 1, "s_per_seq": 0.9, "rep": 0},
    ]
    path = _backend_csv(tmp_path, rows)
    series = load_backend_series(path, max_n=20, skip={"cudaq_v1"}, include_symm=False)
    assert series == [("incremental", [12], [0.01])]


def test_load_backend_series_includes_symm_sector(tmp_path):
    rows = [
        {"benchmark": "dicke", "n_qubits": 12, "backend": "dicke",
         "seq_len": 12, "n_jobs": 1, "s_per_seq": 9.0e-5, "rep": 0},
        {"benchmark": "dicke", "n_qubits": 12, "backend": "dicke",
         "seq_len": 12, "n_jobs": 1, "s_per_seq": 9.4e-5, "rep": 1},
    ]
    path = _backend_csv(tmp_path, rows)
    series = load_backend_series(path, include_symm=True)
    assert series == [("dicke", [12], [9.2e-5])]


def test_load_scaling_series_medians_over_reps_and_sorts_by_n_jobs(tmp_path):
    lightning_rows = [
        {"benchmark": "scaling", "n_jobs": 4, "wall_clock_s": 20.0, "rep": 0},
        {"benchmark": "scaling", "n_jobs": 4, "wall_clock_s": 22.0, "rep": 1},
        {"benchmark": "scaling", "n_jobs": 1, "wall_clock_s": 100.0, "rep": 0},
        {"benchmark": "scaling", "n_jobs": 1, "wall_clock_s": 110.0, "rep": 1},
    ]
    incremental_rows = [
        {"benchmark": "scaling", "n_jobs": 4, "wall_clock_s": 10.0, "rep": 0},
        {"benchmark": "scaling", "n_jobs": 4, "wall_clock_s": 12.0, "rep": 1},
        {"benchmark": "scaling", "n_jobs": 1, "wall_clock_s": 90.0, "rep": 0},
        {"benchmark": "scaling", "n_jobs": 1, "wall_clock_s": 96.0, "rep": 1},
    ]
    l_path = tmp_path / "lightning.csv"
    s_path = tmp_path / "incremental.csv"
    pd.DataFrame(lightning_rows).to_csv(l_path, index=False)
    pd.DataFrame(incremental_rows).to_csv(s_path, index=False)

    n_jobs, lightning_s, incremental_s = load_scaling_series(l_path, s_path)

    assert n_jobs == [1, 4]
    assert lightning_s == [105.0, 21.0]
    assert incremental_s == [93.0, 11.0]


def test_load_scaling_series_raises_on_mismatched_n_jobs(tmp_path):
    l_path = tmp_path / "lightning.csv"
    s_path = tmp_path / "incremental.csv"
    pd.DataFrame(
        [{"benchmark": "scaling", "n_jobs": 1, "wall_clock_s": 100.0, "rep": 0}]
    ).to_csv(l_path, index=False)
    pd.DataFrame(
        [{"benchmark": "scaling", "n_jobs": 4, "wall_clock_s": 10.0, "rep": 0}]
    ).to_csv(s_path, index=False)

    with pytest.raises(ValueError, match="different n_jobs"):
        load_scaling_series(l_path, s_path)
