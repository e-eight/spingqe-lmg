import numpy as np
import pennylane as qml
import pytest
from scipy.linalg import expm

from spingqe_lmg.config import PoolConfig
from spingqe_lmg.enums import Connectivity, PoolKind
from spingqe_lmg.exact import dense_matrix
from spingqe_lmg.pools import (
    DEFAULT_ANGLES,
    build_collective_pool,
    build_pauli_pair_pool,
    build_pool,
)


def test_pool_size_formula_all_to_all():
    n, n_angles = 5, len(DEFAULT_ANGLES)
    pool = build_pauli_pair_pool(n)
    assert len(pool) == 3 * (n * (n - 1) // 2) * n_angles + n * n_angles


def test_pool_size_formula_nn():
    n, n_angles = 5, len(DEFAULT_ANGLES)
    pool = build_pauli_pair_pool(n, connectivity="nn")
    assert len(pool) == 3 * (n - 1) * n_angles + n * n_angles


def test_collective_pool_size():
    pool = build_collective_pool(4)
    assert len(pool) == 3 * len(DEFAULT_ANGLES)


@pytest.mark.parametrize("kind", [PoolKind.PAULI_PAIR, PoolKind.COLLECTIVE])
def test_ops_unitary(kind):
    cfg = PoolConfig(kind=kind, connectivity=Connectivity.ALL, angles=(np.pi / 4, -np.pi / 8))
    pool = build_pool(cfg, 3)
    for op in pool:
        mat = op.matrix()
        assert np.allclose(mat @ mat.conj().T, np.eye(mat.shape[0]), atol=1e-10), op.label


@pytest.mark.parametrize("kind", [PoolKind.PAULI_PAIR, PoolKind.COLLECTIVE])
def test_pool_parity_conserving(kind):
    n = 3
    cfg = PoolConfig(kind=kind, angles=(np.pi / 4,))
    pool = build_pool(cfg, n)
    parity = dense_matrix(qml.Hamiltonian([1.0], [qml.prod(*(qml.PauliZ(i) for i in range(n)))]), n)
    for op in pool:
        mat = op.matrix()
        assert np.linalg.norm(mat @ parity - parity @ mat) < 1e-10, op.label


def test_collective_matches_expm():
    n, theta = 3, 0.37
    pool = build_collective_pool(n, angles=(theta,), generators=("Jx2",))
    jx = 0.5 * sum(dense_matrix(qml.Hamiltonian([1.0], [qml.PauliX(i)]), n) for i in range(n))
    expected = expm(-1j * theta * (jx @ jx))
    assert np.allclose(pool[0].matrix(), expected, atol=1e-10)


def test_pauli_rot_matches_expm():
    n, theta = 2, 0.61
    pool = build_pauli_pair_pool(n, angles=(theta,), paulis=("XX",), include_single_z=False)
    xx = dense_matrix(qml.Hamiltonian([1.0], [qml.PauliX(0) @ qml.PauliX(1)]), n)
    # qml.PauliRot(theta, P) = exp(-i theta P / 2)
    expected = expm(-1j * theta / 2 * xx)
    assert np.allclose(pool[0].matrix(), expected, atol=1e-10)


def test_pool_config_custom_paulis():
    # cross words (XY/YX) are needed for squeezing-type correlations; they must
    # flow from PoolConfig through build_pool, and they still conserve parity
    # (any two-qubit Pauli word picks up two signs under the global Z flip)
    n = 3
    cfg = PoolConfig(kind=PoolKind.PAULI_PAIR, paulis=("XY", "YX"), angles=(0.3,))
    pool = build_pool(cfg, n)
    words = {op.pauli_word for op in pool if len(op.wires) == 2}
    assert words == {"XY", "YX"}
    parity = dense_matrix(qml.Hamiltonian([1.0], [qml.prod(*(qml.PauliZ(i) for i in range(n)))]), n)
    for op in pool:
        mat = op.matrix()
        assert np.linalg.norm(mat @ parity - parity @ mat) < 1e-10, op.label


def test_angle_scale_rescales_pool_angles():
    n, scale = 16, 8.0
    base = (np.pi / 2, -np.pi / 4)
    cfg = PoolConfig(kind=PoolKind.COLLECTIVE, angles=base, angle_scale=scale)
    pool = build_pool(cfg, n)
    got = sorted({op.angle for op in pool})
    expected = sorted(a * scale / n for a in base)
    assert got == pytest.approx(expected)


def test_angle_scale_zero_is_identity():
    cfg_off = PoolConfig(kind=PoolKind.PAULI_PAIR, angles=(0.3, -0.7))
    cfg_zero = PoolConfig(kind=PoolKind.PAULI_PAIR, angles=(0.3, -0.7), angle_scale=0.0)
    assert build_pool(cfg_off, 4) == build_pool(cfg_zero, 4)


def test_pool_labels_unique():
    pool = build_pauli_pair_pool(4)
    labels = [op.label for op in pool]
    assert len(labels) == len(set(labels))


def test_pool_n1_produces_no_pairs():
    """N=1 produces 0 Pauli-pair operators regardless of connectivity."""
    from spingqe_lmg.enums import Connectivity

    for conn in (Connectivity.ALL, Connectivity.NN):
        pool = build_pauli_pair_pool(1, connectivity=conn)
        pairs = [op for op in pool if len(op.wires) == 2]
        assert len(pairs) == 0


def test_pool_n2_nn_equals_all():
    """At N=2, nearest-neighbor and all-to-all produce the same pool."""
    from spingqe_lmg.enums import Connectivity

    pool_all = build_pauli_pair_pool(2, connectivity=Connectivity.ALL)
    pool_nn = build_pauli_pair_pool(2, connectivity=Connectivity.NN)
    assert len(pool_all) == len(pool_nn)
    assert [op.label for op in pool_all] == [op.label for op in pool_nn]
