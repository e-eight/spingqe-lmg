import numpy as np
import pytest

cudaq = pytest.importorskip("cudaq")

from spingqe_lmg.evaluators.cudaq import CudaQEvaluator  # noqa: E402
from spingqe_lmg.evaluators.pennylane import PennyLaneEvaluator, initial_energy  # noqa: E402
from spingqe_lmg.hamiltonians import (  # noqa: E402
    heisenberg_xxz_hamiltonian,
    lmg_hamiltonian,
    lmg_mean_field_angle,
)
from spingqe_lmg.pools import (  # noqa: E402
    build_collective_pool,
    build_pauli_pair_pool,
)

pytestmark = pytest.mark.hardware


def parity_check(pool, ham, n, init_angle=0.0, n_seq=4, seq_len=5, seed=0):
    """CUDA-Q and PennyLane evaluators must agree on per-step energies."""
    rng = np.random.default_rng(seed)
    idx_batch = rng.integers(0, len(pool), size=(n_seq, seq_len))
    pl = PennyLaneEvaluator(pool, ham, n, init_angle=init_angle)
    cq = CudaQEvaluator(pool, ham, n, init_angle=init_angle)
    a, b = pl(idx_batch), cq(idx_batch)
    np.testing.assert_allclose(a, b, atol=1e-8)


def test_parity_pairwise_lmg():
    n = 3
    ham = lmg_hamiltonian(n, h=1.0, lam=0.8)
    parity_check(build_pauli_pair_pool(n), ham, n)


def test_parity_extended_words():
    n = 3
    ham = lmg_hamiltonian(n, h=1.3, lam=1.5, gamma=0.4)
    pool = build_pauli_pair_pool(n, paulis=("ZZ", "XX", "YY", "YZ", "XY"))
    parity_check(pool, ham, n, seed=1)


def test_parity_collective():
    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=1.2)
    parity_check(build_collective_pool(n), ham, n, seed=2)


def test_parity_mean_field_reference():
    n = 3
    h, lam = 1.0, 1.8
    ham = lmg_hamiltonian(n, h=h, lam=lam)
    theta = lmg_mean_field_angle(h, lam)
    parity_check(build_pauli_pair_pool(n), ham, n, init_angle=theta, seed=3)


def test_parity_heisenberg():
    # asymmetric couplings catch sign-convention errors that symmetric LMG hides
    n = 3
    ham = heisenberg_xxz_hamiltonian(n, j1=10.0, j2=10.0)
    parity_check(build_pauli_pair_pool(n, connectivity="nn"), ham, n, seed=4)


def test_empty_circuit_energy_matches_initial():
    n = 3
    ham = lmg_hamiltonian(n, h=1.0, lam=0.5)
    pool = build_pauli_pair_pool(n, angles=(1e-14,), paulis=("ZZ",), include_single_z=False)
    cq = CudaQEvaluator(pool, ham, n)
    es = cq(np.array([[0]]))
    assert es[0, 0] == pytest.approx(initial_energy(ham, n), abs=1e-9)


def test_chain_states_matches_prefix_path():
    # v2 (state-handle chaining) must agree with v1 (one prefix circuit per
    # step) at machine precision: both run on the same simulator
    n = 4
    ham = lmg_hamiltonian(n, h=1.3, lam=1.5, gamma=0.4)
    pool = build_pauli_pair_pool(n, paulis=("ZZ", "XX", "YY", "YZ", "XY"))
    rng = np.random.default_rng(5)
    idx_batch = rng.integers(0, len(pool), size=(4, 6))
    v2 = CudaQEvaluator(pool, ham, n)
    v1 = CudaQEvaluator(pool, ham, n, chain_states=False)
    np.testing.assert_allclose(v2(idx_batch), v1(idx_batch), atol=1e-12)


def test_chain_states_matches_prefix_path_collective():
    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=1.2)
    pool = build_collective_pool(n)
    rng = np.random.default_rng(6)
    idx_batch = rng.integers(0, len(pool), size=(3, 5))
    v2 = CudaQEvaluator(pool, ham, n)
    v1 = CudaQEvaluator(pool, ham, n, chain_states=False)
    np.testing.assert_allclose(v2(idx_batch), v1(idx_batch), atol=1e-12)


@pytest.mark.gpu
def test_parity_on_nvidia_target():
    # runs with the Phase A benchmark/smoke job: v2 on the nvidia target must
    # match PennyLane (fp32 statevector -> looser tolerance than qpp-cpu)
    n = 6
    ham = lmg_hamiltonian(n, h=1.0, lam=1.5)
    pool = build_pauli_pair_pool(n, paulis=("ZZ", "XX", "YY", "YZ", "XY"))
    rng = np.random.default_rng(7)
    idx_batch = rng.integers(0, len(pool), size=(3, 8))
    cq = CudaQEvaluator(pool, ham, n)
    if cq.target != "nvidia":
        pytest.skip("no CUDA device")
    pl = PennyLaneEvaluator(pool, ham, n)
    np.testing.assert_allclose(pl(idx_batch), cq(idx_batch), atol=1e-4)


def test_cache_reuse():
    n = 3
    ham = lmg_hamiltonian(n, h=1.0, lam=0.5)
    pool = build_pauli_pair_pool(n)
    cq = CudaQEvaluator(pool, ham, n)
    idx = np.array([[0, 1]])
    result1 = cq(idx)
    import time

    t0 = time.perf_counter()
    result2 = cq(idx)
    t1 = time.perf_counter()
    np.testing.assert_array_equal(result1, result2)
    assert (t1 - t0) < 1e-3, f"cache hit took {t1 - t0:.4f}s — cache may not be working"


def test_cudaq_initial_energy_matches_standalone():
    """Finding 9: CudaQEvaluator.initial_energy must match standalone helper."""
    from spingqe_lmg.evaluators.pennylane import initial_energy

    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=0.5)
    pool = build_pauli_pair_pool(n)

    ev = CudaQEvaluator(pool, ham, n)

    e_init = ev.initial_energy()
    assert isinstance(e_init, float)

    ref = initial_energy(ham, n, 0.0)
    assert e_init == pytest.approx(ref, abs=1e-6)


def test_cudaq_cache_uses_lru():
    """Finding 10: CudaQEvaluator must use LRUCache with eviction."""
    from spingqe_lmg.core.caching import LRUCache

    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=0.5)
    pool = build_pauli_pair_pool(n)

    ev = CudaQEvaluator(pool, ham, n, cache_maxsize=2)
    assert isinstance(ev._cache, LRUCache)
    assert ev._cache.maxsize == 2

    rng = np.random.default_rng(1)
    keys = [tuple(int(x) for x in rng.integers(0, len(pool), size=3)) for _ in range(3)]
    for k in keys:
        ev(np.array([k]))

    assert len(ev._cache) == 2
