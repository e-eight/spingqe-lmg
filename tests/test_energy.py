import numpy as np
import pytest

from spingqe_lmg.evaluators.pennylane import PennyLaneEvaluator, initial_energy, sequence_energies
from spingqe_lmg.hamiltonians import lmg_hamiltonian
from spingqe_lmg.pools import build_pauli_pair_pool


@pytest.fixture
def setup():
    n = 3
    ham = lmg_hamiltonian(n, h=1.0, lam=0.8, gamma=0.0)
    pool = build_pauli_pair_pool(n)
    return n, ham, pool


def test_initial_energy_is_minus_nh_over_2(setup):
    n, ham, _ = setup
    # the XX terms have zero expectation in |0...0>, so <H> = -n h / 2 even at lam != 0
    assert initial_energy(ham, n) == pytest.approx(-n / 2, abs=1e-12)


def test_empty_sequence(setup):
    n, ham, _ = setup
    assert sequence_energies((), ham, n).shape == (0,)


def test_prefix_energies_match_truncations(setup):
    n, ham, pool = setup
    rng = np.random.default_rng(3)
    idx = rng.integers(0, len(pool), size=6)
    ops = tuple(pool[i] for i in idx)
    full = sequence_energies(ops, ham, n)
    assert full.shape == (6,)
    for k in range(1, 7):
        truncated = sequence_energies(ops[:k], ham, n)
        assert truncated[-1] == pytest.approx(full[k - 1], abs=1e-10)


def test_single_op_energy_manual():
    # N=2, H = -(h/2)(Z0+Z1) - (lam/4) * 2 * X0 X1; apply XX(theta) to |00>:
    # |psi> = cos(theta/2)|00> - i sin(theta/2)|11>
    # <Z0+Z1> = 2 cos(theta), <X0X1> = -2 cos(theta/2) sin(theta/2) * ... compute directly
    h, lam, theta = 1.0, 0.8, np.pi / 3
    ham = lmg_hamiltonian(2, h=h, lam=lam, gamma=0.0)
    pool = build_pauli_pair_pool(2, angles=(theta,), paulis=("XX",), include_single_z=False)
    (op,) = pool
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    psi = np.array([c, 0, 0, -1j * s])  # basis |00>,|01>,|10>,|11>
    z_sum = 2 * (c**2) - 2 * (s**2)
    x0x1 = float(np.real(psi.conj() @ np.kron([[0, 1], [1, 0]], [[0, 1], [1, 0]]) @ psi))
    expected = -(h / 2) * z_sum - (lam / 4) * x0x1
    got = sequence_energies((op,), ham, 2)[0]
    assert got == pytest.approx(expected, abs=1e-10)


def test_collective_sequence_unitary_in_circuit():
    # regression: building collective generators inside the QNode used to queue
    # orphaned Sum/Prod composites, which the device applied as non-unitary
    # gates — energies exploded exponentially with sequence length
    from spingqe_lmg.pools import build_collective_pool

    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=0.5)
    pool = build_collective_pool(n, angles=(np.pi / 4,))
    jz, jx2, jy2 = pool
    bound = sum(abs(c) for c in ham.terms()[0])  # |<H>| <= sum |coeffs|
    es = sequence_energies((jx2, jy2, jz, jx2, jy2), ham, n)
    assert np.all(np.abs(es) <= bound + 1e-9)


def test_collective_jz_preserves_basis_state_energy():
    # exp(-i theta Jz) only phases |0...0>, so every prefix energy equals E_init
    from spingqe_lmg.pools import build_collective_pool

    n = 3
    ham = lmg_hamiltonian(n, h=1.0, lam=0.8)
    pool = build_collective_pool(n, angles=(0.7,), generators=("Jz",))
    es = sequence_energies((pool[0], pool[0]), ham, n)
    assert np.allclose(es, initial_energy(ham, n), atol=1e-10)


def test_collective_prefix_matches_truncations():
    from spingqe_lmg.pools import build_collective_pool

    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=1.2)
    pool = build_collective_pool(n)
    rng = np.random.default_rng(9)
    ops = tuple(pool[i] for i in rng.integers(0, len(pool), size=5))
    full = sequence_energies(ops, ham, n)
    for k in range(1, 6):
        truncated = sequence_energies(ops[:k], ham, n)
        assert truncated[-1] == pytest.approx(full[k - 1], abs=1e-10)


def test_collective_single_gate_matches_expm():
    # the same check test_collective_matches_expm does on matrices, but through
    # an actual circuit execution
    import pennylane as qml
    from scipy.linalg import expm

    from spingqe_lmg.pools import build_collective_pool

    n, theta = 3, 0.37
    ham = lmg_hamiltonian(n, h=1.0, lam=0.5)
    pool = build_collective_pool(n, angles=(theta,), generators=("Jx2",))
    psi0 = np.zeros(2**n)
    psi0[0] = 1.0
    jx = 0.5 * sum(
        qml.matrix(qml.Hamiltonian([1.0], [qml.PauliX(i)]), wire_order=range(n)) for i in range(n)
    )
    psi = expm(-1j * theta * (jx @ jx)) @ psi0
    ham_mat = qml.matrix(ham, wire_order=range(n))
    expected = float(np.real(psi.conj() @ ham_mat @ psi))
    got = sequence_energies((pool[0],), ham, n)[0]
    assert got == pytest.approx(expected, abs=1e-10)


def test_batch_matches_loop(setup):
    n, ham, pool = setup
    rng = np.random.default_rng(5)
    idx_batch = rng.integers(0, len(pool), size=(4, 5))
    serial = PennyLaneEvaluator(pool, ham, n, n_jobs=1)
    parallel = PennyLaneEvaluator(pool, ham, n, n_jobs=2)
    res_serial = serial(idx_batch)
    res_parallel = parallel(idx_batch)
    loop = np.stack([sequence_energies(tuple(pool[i] for i in row), ham, n) for row in idx_batch])
    assert np.allclose(res_serial, loop)
    assert np.allclose(res_parallel, loop)


def test_cache_hit_returns_same(setup):
    n, ham, pool = setup
    ev = PennyLaneEvaluator(pool, ham, n)
    idx = np.array([[0, 1, 2]])
    result1 = ev(idx)
    result2 = ev(idx)
    np.testing.assert_array_equal(result1, result2)


def test_sequence_energies_output_length():
    """N gates produce N energy values (one after each gate)."""
    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=1.5)
    pool = build_pauli_pair_pool(n, angles=(0.1,), paulis=("XX",), include_single_z=False)
    for ngates in range(1, 6):
        ops = tuple(pool[i % len(pool)] for i in range(ngates))
        es = sequence_energies(ops, ham, n)
        assert len(es) == ngates, f"ngates={ngates} produced len={len(es)}"


def test_sequence_energies_first_is_not_initial():
    """es[0] is energy after first gate, not the initial state energy."""
    n = 3
    ham = lmg_hamiltonian(n, h=1.0, lam=0.8, gamma=0.0)
    pool = build_pauli_pair_pool(n, angles=(0.1,), paulis=("XX",), include_single_z=False)
    e_init = initial_energy(ham, n)
    es = sequence_energies((pool[0], pool[1]), ham, n)
    assert es[0] != pytest.approx(e_init, abs=1e-9), "es[0] should be after gate 1, not initial"


def test_sequence_energies_includes_initial_energy_via_initial_energy_fn():
    """initial_energy() returns snap[0]; sequence_energies starts from snap[1].
    They differ: initial_energy gives the state-prep energy, sequence_energies
    gives post-gate energies. Both are needed by different callers."""
    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=1.5)
    pool = build_pauli_pair_pool(n, angles=(0.1,), paulis=("XX",), include_single_z=False)
    e_init = initial_energy(ham, n)
    es = sequence_energies((pool[0],), ham, n)
    assert len(es) == 1
    assert es[0] != pytest.approx(e_init, abs=1e-9), (
        "sequence_energies returns post-gate energies, not initial energy"
    )


def test_build_offdiag_perms_rejects_odd_y_terms():
    """build_offdiag_perms must raise NotImplementedError for num_y odd."""
    from spingqe_lmg.core.parity import build_offdiag_perms

    n_qubits = 4
    half = 2 ** (n_qubits - 1)
    wire0_bit = np.array([bin(k).count("1") % 2 for k in range(half)], dtype=np.int8)

    odd_y_terms = [
        (1.0, [(0, "Y")]),  # one Y on wire 0 -> num_y = 1
    ]

    with pytest.raises(NotImplementedError, match="odd.*Y"):
        build_offdiag_perms(odd_y_terms, n_qubits, wire0_bit)
