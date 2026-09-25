"""Tests for PennyLaneGroupedEvaluator — final-energy agreement and edge cases."""

import numpy as np
import pytest

from spingqe_lmg.evaluators.pennylane import (
    PennyLaneEvaluator,
    PennyLaneGroupedEvaluator,
)
from spingqe_lmg.hamiltonians import lmg_hamiltonian
from spingqe_lmg.pools import build_pauli_pair_pool


@pytest.fixture
def setup():
    n = 6
    ham = lmg_hamiltonian(n, h=1.0, lam=1.5, gamma=0.0)
    pool = build_pauli_pair_pool(n)
    return n, ham, pool


def test_grouped_matches_standard_final_energy(setup):
    """Grouped evaluator's final energy matches standard evaluator at each step."""
    n, ham, pool = setup
    rng = np.random.default_rng(17)
    idx = rng.integers(0, len(pool), size=(4, 5))

    standard = PennyLaneEvaluator(pool, ham, n)
    grouped = PennyLaneGroupedEvaluator(pool, ham, n)

    std_result = standard(idx)   # shape (4, 5) — per-step energies
    grp_result = grouped(idx)    # shape (4, 5) — final energy broadcast

    # The grouped evaluator returns the same final energy for all steps.
    # Standard evaluator's last column is the final energy.
    for b in range(4):
        expected_final = std_result[b, -1]
        got_final = grp_result[b, -1]
        assert got_final == pytest.approx(float(expected_final), rel=1e-8, abs=1e-10)


def test_grouped_empty_sequence():
    """Zero-length sequence returns empty per-step energies."""
    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=0.8)
    pool = build_pauli_pair_pool(n)

    grouped = PennyLaneGroupedEvaluator(pool, ham, n)
    result = grouped(np.empty((1, 0), dtype=int))
    assert result.shape == (1, 0)


def test_grouped_initial_energy(setup):
    """initial_energy() inherits from parent — sanity check it runs."""
    n, ham, pool = setup
    grouped = PennyLaneGroupedEvaluator(pool, ham, n)
    e0 = grouped.initial_energy()
    assert e0 == pytest.approx(-n / 2, abs=1e-10)


def test_grouped_cache_hit(setup):
    """Cache returns identical arrays on repeat calls."""
    n, ham, pool = setup
    grouped = PennyLaneGroupedEvaluator(pool, ham, n)
    idx = np.array([[0, 1, 2, 3, 4]])
    result1 = grouped(idx)
    result2 = grouped(idx)
    np.testing.assert_array_equal(result1, result2)


@pytest.mark.gpu
def test_grouped_gpu_device_construction():
    """PennyLaneGroupedEvaluator can be constructed with device_name='lightning.gpu'.

    Does not evaluate — only verifies that the grouped observable
    partitioning succeeds on a GPU device context.  Skipped when CUDA
    is not available.
    """
    pytest.importorskip("pennylane_lightning_gpu")
    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=1.5)
    pool = build_pauli_pair_pool(n)

    grouped = PennyLaneGroupedEvaluator(pool, ham, n, device_name="lightning.gpu")
    idx = np.array([[0, 1, 2, 3]])
    result = grouped(idx)
    assert result.shape == (1, 4)
