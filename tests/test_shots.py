import numpy as np
import pytest

from spingqe_lmg.evaluators.pennylane import PennyLaneEvaluator, sequence_energies
from spingqe_lmg.exact import dicke_matrix, ground_state
from spingqe_lmg.hamiltonians import lmg_hamiltonian
from spingqe_lmg.pools import build_pauli_pair_pool
from spingqe_lmg.refine import refine_angles, refine_angles_spsa


@pytest.fixture
def setup():
    n = 3
    ham = lmg_hamiltonian(n, h=1.0, lam=0.8)
    pool = build_pauli_pair_pool(n)
    return n, ham, pool


def test_shot_energies_statistically_consistent(setup):
    n, ham, pool = setup
    ops = tuple(pool[i] for i in (3, 17, 25, 8))
    exact = sequence_energies(ops, ham, n)
    sampled = sequence_energies(ops, ham, n, shots=8192, seed=1)
    # sum of |coeffs| bounds the per-shot spread; 8192 shots => sigma <~ 0.03
    assert np.all(np.abs(sampled - exact) < 0.15)
    assert np.any(sampled != exact)  # actually sampled, not analytic


def test_shot_energies_seeded_reproducible(setup):
    n, ham, pool = setup
    ops = tuple(pool[i] for i in (3, 17, 25))
    a = sequence_energies(ops, ham, n, shots=256, seed=7)
    b = sequence_energies(ops, ham, n, shots=256, seed=7)
    np.testing.assert_array_equal(a, b)


def test_evaluator_threads_shots(setup):
    n, ham, pool = setup
    idx = np.array([[3, 17, 25], [3, 17, 25]])
    noisy = PennyLaneEvaluator(pool, ham, n, shots=256, seed=3)
    exact = PennyLaneEvaluator(pool, ham, n)
    out_noisy, out_exact = noisy(idx), exact(idx)
    assert not np.allclose(out_noisy, out_exact)
    # cache: repeated sequence reuses the same noisy draw
    assert np.array_equal(out_noisy[0], out_noisy[1])


def test_cobyla_refines_under_shots():
    # comparison path: COBYLA under shots runs and improves on the raw
    # structure (its noise-degradation relative to SPSA is measured in the
    # shots study, not asserted here)
    from spingqe_lmg.refine import refine_angles_cobyla

    n, h, lam = 2, 1.0, 0.5
    ham = lmg_hamiltonian(n, h=h, lam=lam)
    pool = build_pauli_pair_pool(n, angles=(np.pi / 8,))
    by_label = {op.label: op for op in pool}
    ops = [by_label["XX(0,1,+0.3927)"], by_label["Z(0,+0.3927)"], by_label["XX(0,1,+0.3927)"]]
    raw = refine_angles(ops, ham, n).initial_energy
    result = refine_angles_cobyla(ops, ham, n, shots=2048, seed=5)
    assert result.energy <= raw + 1e-9
    assert result.n_evaluations <= 602


def test_spsa_refines_under_shots():
    # N=2 ground-reachable structure (XX-Z-XX); SPSA under shots should get
    # close to the exact ground energy, judged by the analytic energy of its
    # final angles
    n, h, lam = 2, 1.0, 0.5
    ham = lmg_hamiltonian(n, h=h, lam=lam)
    e0 = ground_state(dicke_matrix(n, h, lam))[0]
    pool = build_pauli_pair_pool(n, angles=(np.pi / 8,))
    by_label = {op.label: op for op in pool}
    ops = [by_label["XX(0,1,+0.3927)"], by_label["Z(0,+0.3927)"], by_label["XX(0,1,+0.3927)"]]
    exact_start = refine_angles(ops, ham, n)  # analytic optimum for comparison
    result = refine_angles_spsa(ops, ham, n, shots=2048, seed=5, iterations=250)
    assert result.energy <= exact_start.initial_energy  # improved on the raw structure
    assert result.energy - e0 < 0.05  # near-ground despite noisy optimization
    assert result.n_evaluations == 1 + 2 * 250
