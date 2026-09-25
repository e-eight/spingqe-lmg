import numpy as np
import pytest

from spingqe_lmg.exact import (
    collective_spin_matrices,
    dense_matrix,
    dicke_matrix,
    exact_qpt_curve,
    ground_state,
    order_parameter_exact,
)
from spingqe_lmg.hamiltonians import lmg_hamiltonian


@pytest.mark.slow
@pytest.mark.parametrize("n", range(2, 11))
@pytest.mark.parametrize("lam_over_h", [0.0, 0.5, 1.0, 1.5, 2.0])
@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
def test_dicke_matches_dense(n, lam_over_h, gamma):
    h = 1.0
    lam = lam_over_h * h
    e_dicke = ground_state(dicke_matrix(n, h, lam, gamma))[0]
    e_dense = ground_state(dense_matrix(lmg_hamiltonian(n, h=h, lam=lam, gamma=gamma), n))[0]
    assert abs(e_dicke - e_dense) < 1e-10


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_lambda_zero_limit(n):
    h = 1.7
    assert abs(ground_state(dicke_matrix(n, h, 0.0, 0.0))[0] - (-n * h / 2)) < 1e-12


@pytest.mark.parametrize("n", [2, 4, 8])
def test_h_zero_gamma_zero_limit(n):
    # H = -(lam/4N) sum_{i!=j} X_i X_j; ground state all spins aligned in x:
    # E0 = -(lam/N) jx_max^2 + lam/4 = -lam(N-1)/4
    lam = 2.3
    assert abs(ground_state(dicke_matrix(n, 0.0, lam, 0.0))[0] - (-lam * (n - 1) / 4)) < 1e-10


def test_su2_algebra():
    jx, jy, jz = collective_spin_matrices(6)
    assert np.linalg.norm(jx @ jy - jy @ jx - 1j * jz) < 1e-12
    assert np.linalg.norm(jy @ jz - jz @ jy - 1j * jx) < 1e-12
    assert np.linalg.norm(jz @ jx - jx @ jz - 1j * jy) < 1e-12


def test_qpt_energy_monotone_in_lambda():
    lams = np.linspace(0.0, 2.0, 21)
    curve = exact_qpt_curve(10, 1.0, lams, 0.0)
    assert np.all(np.diff(curve) < 1e-12)  # E0 decreases with coupling


def test_order_parameter_grows_across_qpt():
    weak = order_parameter_exact(10, 1.0, 0.2)
    strong = order_parameter_exact(10, 1.0, 2.0)
    assert weak < 0.2
    assert strong > 0.5


def test_dicke_scales_to_large_n():
    # the whole point of the Dicke route: N = 1000 in well under a second
    e0 = ground_state(dicke_matrix(1000, 1.0, 2.0, 0.0))[0]
    assert e0 < -500.0


def test_dicke_matrix_requires_n_ge_2():
    with pytest.raises(ValueError, match="n_qubits must be >= 2"):
        dicke_matrix(0, h=1.0, lam=0.5)
    with pytest.raises(ValueError, match="n_qubits must be >= 2"):
        dicke_matrix(1, h=1.0, lam=0.5)


def test_ground_state_matches_eigh():
    matrix = dicke_matrix(4, h=1.0, lam=1.5, gamma=0.3)
    e0, psi0 = ground_state(matrix)

    vals, vecs = np.linalg.eigh(matrix)
    assert e0 == pytest.approx(vals[0])
    np.testing.assert_allclose(psi0, vecs[:, 0])
    assert np.linalg.norm(psi0) == pytest.approx(1.0)
    np.testing.assert_allclose(matrix @ psi0, e0 * psi0, atol=1e-10)


def test_order_parameter_exact_requires_n_ge_2():
    from spingqe_lmg.exact import order_parameter_exact

    with pytest.raises(ValueError, match="n_qubits must be >= 2"):
        order_parameter_exact(0, h=1.0, lam=0.5)
    with pytest.raises(ValueError, match="n_qubits must be >= 2"):
        order_parameter_exact(1, h=1.0, lam=0.5)
