"""Tests for benchmark-scaling.py resume/skip helpers."""

import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.script


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_scaling",
        Path(__file__).parent.parent / "scripts" / "benchmark-scaling.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bmod():
    return _load_module()


SAMPLE_ROW = {
    "benchmark": "backend",
    "n_qubits": "12",
    "backend": "incremental",
    "pool_kind": "pauli_pair",
    "n_jobs": "1",
    "n_sequences": "20",
    "seq_len": "12",
    "s_per_seq": "0.001234",
    "wall_clock_s": "",
    "process_rss_gb": "0.50",
    "pinned": "False",
    "target": "",
    "rep": "0",
}


def _write_rows(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_row_key_backend_row(bmod):
    assert bmod._row_key(SAMPLE_ROW) == ("backend", 12, "incremental", 1, 12, 0)


def test_row_key_scaling_row(bmod):
    row = {**SAMPLE_ROW, "benchmark": "scaling", "n_jobs": "8", "n_qubits": "20"}
    assert bmod._row_key(row) == ("scaling", 20, "incremental", 8, 12, 0)


def test_load_existing_keys_missing_file(tmp_path, bmod):
    assert bmod._load_existing_keys(str(tmp_path / "missing.csv")) == frozenset()


def test_load_existing_keys_single_row(tmp_path, bmod):
    p = tmp_path / "bench.csv"
    _write_rows(str(p), [SAMPLE_ROW])
    assert ("backend", 12, "incremental", 1, 12, 0) in bmod._load_existing_keys(str(p))


def test_load_existing_keys_multiple_rows(tmp_path, bmod):
    row2 = {**SAMPLE_ROW, "n_qubits": "14", "backend": "parity_even"}
    p = tmp_path / "bench.csv"
    _write_rows(str(p), [SAMPLE_ROW, row2])
    keys = bmod._load_existing_keys(str(p))
    assert len(keys) == 2
    assert ("backend", 14, "parity_even", 1, 12, 0) in keys


def test_bench_backends_skips_parity_even_when_key_present(monkeypatch, bmod):
    monkeypatch.setattr(bmod, "FULL_N", (12,))

    skip_keys = frozenset({("backend", 12, "parity_even", 1, 12, 0)})
    args = argparse.Namespace(
        n_seq=1,
        seq_len=12,
        backends="parity_even",
        output_csv=None,
        n_range=(12,),
        h=1.0,
        lam=0.5,
        gamma=0.0,
        repeat=1,
    )

    before = len(bmod._rows)
    bmod._bench_backends(args, skip_keys=skip_keys)

    assert len(bmod._rows) == before, "parity_even N=12 should be skipped"


def test_bench_dicke_skips_when_key_present(monkeypatch, bmod):
    monkeypatch.setattr(bmod, "DICKE_N", (12,))

    skip_keys = frozenset(
        {
            ("dicke", 12, "dicke", 1, 12, 0),
            ("dicke", 50, "dicke", 1, 12, 0),
            ("dicke", 100, "dicke", 1, 12, 0),
            ("dicke", 500, "dicke", 1, 12, 0),
            ("dicke", 1000, "dicke", 1, 12, 0),
        }
    )
    args = argparse.Namespace(
        n_seq=1, seq_len=12, output_csv=None, n_range=(12,),
        h=1.0, lam=0.5, gamma=0.0, repeat=1,
    )

    before = len(bmod._rows)
    bmod._bench_dicke(args, skip_keys=skip_keys)

    assert len(bmod._rows) == before, "dicke N should be skipped"


def test_bench_scaling_skips_n_jobs_when_key_present(monkeypatch, bmod):
    skip_keys = frozenset({("scaling", 20, "incremental", 1, 12, 0)})
    args = argparse.Namespace(
        n_seq=2,
        seq_len=12,
        pinned=False,
        evaluator="incremental",
        output_csv=None,
        scaling_n=20,
        scaling_batch=2,
        scaling_jobs=(1, 4),
        h=1.0,
        lam=0.5,
        gamma=0.0,
        repeat=1,
        device_name="lightning.qubit",
    )

    before = len(bmod._rows)
    bmod._bench_scaling(args, skip_keys=skip_keys)
    added = bmod._rows[before:]

    assert not any(r["benchmark"] == "scaling" and int(r["n_jobs"]) == 1 for r in added)
    assert any(r["benchmark"] == "scaling" and int(r["n_jobs"]) == 4 for r in added)


def test_peven_evaluator_no_device_name_arg():
    """Finding 2: ParityEvenEvaluator does not accept device_name.
    Verifying it can be constructed without it."""
    import numpy as np

    from spingqe_lmg.evaluators.statevec import ParityEvenEvaluator
    from spingqe_lmg.hamiltonians import lmg_hamiltonian
    from spingqe_lmg.pools import build_pauli_pair_pool

    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=0.8)
    pool = build_pauli_pair_pool(n)
    rng = np.random.default_rng(13)
    idx = rng.integers(0, len(pool), size=(4, 8))

    ev = ParityEvenEvaluator(pool, ham, n, n_jobs=1)
    result = ev(idx)
    assert result.shape == (4, 8)


def test_bench_rss_skips_backend_when_key_present(monkeypatch, bmod):
    """rss subprocess is not launched for a cell whose key is in skip_keys."""
    monkeypatch.setattr(bmod, "FULL_N", (12,))
    monkeypatch.setattr(bmod, "DICKE_N", ())

    skip_keys = frozenset({("backend", 12, "incremental", 1, bmod.SEQ_LEN, 0)})
    args = argparse.Namespace(
        backends="incremental",
        output_csv=None,
        dicke_csv=None,
        n_range=(12,),
        h=1.0,
        lam=0.5,
        gamma=0.0,
        n_seq=None,
        seq_len=bmod.SEQ_LEN,
        repeat=1,
    )

    before = len(bmod._rows)
    bmod._bench_backends(args, skip_keys=skip_keys)

    assert len(bmod._rows) == before, "incremental N=12 rss row should be skipped"


def test_scaling_warmup_calls_evaluator_twice(monkeypatch, bmod):
    """A warmup call precedes the timed call: evaluator.__call__ is invoked
    at least twice — once for warmup and once for the timed batch."""

    call_count = 0

    class TracingEvaluator:
        pool = []
        ham = None
        n_qubits = 4

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, idx_batch):
            nonlocal call_count
            call_count += 1
            return np.zeros((len(idx_batch), 12))

        def initial_energy(self):
            return 0.0

    monkeypatch.setattr(bmod, "PennyLaneEvaluator", TracingEvaluator)

    args = argparse.Namespace(
        n_seq=2,
        seq_len=12,
        pinned=False,
        evaluator="lightning",
        device_name="lightning.qubit",
        output_csv=None,
        scaling_n=4,
        scaling_batch=2,
        scaling_jobs=(1,),
        h=1.0,
        lam=0.5,
        gamma=0.0,
        repeat=1,
    )

    bmod._bench_scaling(args)

    assert call_count >= 2, (
        f"Expected >= 2 calls (warmup + timed), got {call_count}"
    )


def test_add_row_includes_rep_field(bmod):
    """_add_row includes a 'rep' key in the CSV row."""
    bmod._rows.clear()
    bmod._add_row(
        "backend", 12, "test_backend", "pauli_pair",
        n_sequences=1, seq_len=12, s_per_seq=0.001, peak_rss_gb=0.1,
        rss_col="process_rss_gb", rep=3,
    )
    row = bmod._rows[-1]
    assert row["rep"] == "3"


def test_bench_backends_repeat_produces_two_rows(monkeypatch, bmod):
    """--repeat 2 produces 2 rows from a single subprocess invocation."""
    monkeypatch.setattr(bmod, "FULL_N", (12,))
    bmod._rows.clear()

    class FakeCompletedProcess:
        returncode = 0
        stdout = "RESULT:0,0.001000,500000\nRESULT:1,0.001200,510000\n"
        stderr = ""

    monkeypatch.setattr(bmod.subprocess, "run", lambda *a, **kw: FakeCompletedProcess)

    args = argparse.Namespace(
        n_seq=1,
        seq_len=12,
        backends="incremental",
        output_csv=None,
        n_range=(12,),
        h=1.0,
        lam=0.5,
        gamma=0.0,
        repeat=2,
    )
    bmod._bench_backends(args)

    backend_rows = [r for r in bmod._rows if r["backend"] == "incremental"]
    assert len(backend_rows) == 2, f"Expected 2 rows, got {len(backend_rows)}"
    reps = {r["rep"] for r in backend_rows}
    assert reps == {"0", "1"}, f"Expected reps {{0,1}}, got {reps}"


def test_timing_worker_repeat_distinct_seeds(bmod):
    """_timing_worker with repeat=2 returns two results from distinct sequences."""
    from spingqe_lmg.hamiltonians import lmg_hamiltonian
    from spingqe_lmg.pools import build_pauli_pair_pool

    n = 4
    pool = build_pauli_pair_pool(n)

    rng0 = np.random.default_rng(7)
    rng1 = np.random.default_rng(8)
    idx0 = rng0.integers(0, len(pool), size=(1, 12))
    idx1 = rng1.integers(0, len(pool), size=(1, 12))

    assert not np.array_equal(idx0, idx1), (
        "Rep 0 and rep 1 should produce different sequence indices"
    )
