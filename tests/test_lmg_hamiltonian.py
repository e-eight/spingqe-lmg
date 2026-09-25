import numpy as np
import pennylane as qml
import pytest

from spingqe_lmg.exact import dense_matrix
from spingqe_lmg.hamiltonians import heisenberg_xxz_hamiltonian, lmg_hamiltonian


def test_term_count_gamma_zero():
    n = 5
    ham = lmg_hamiltonian(n, h=1.0, lam=0.7, gamma=0.0)
    coeffs, _ = ham.terms()
    assert len(coeffs) == n + n * (n - 1) // 2


def test_term_count_gamma_nonzero():
    n = 5
    ham = lmg_hamiltonian(n, h=1.0, lam=0.7, gamma=0.3)
    coeffs, _ = ham.terms()
    assert len(coeffs) == n + 2 * (n * (n - 1) // 2)


def test_coefficients():
    n = 4
    h, lam, gamma = 1.3, 0.9, 0.5
    ham = lmg_hamiltonian(n, h=h, lam=lam, gamma=gamma)
    coeffs, ops = ham.terms()
    for c, op in zip(coeffs, ops, strict=True):
        name = "".join(sorted({type(o).__name__ for o in getattr(op, "operands", [op])}))
        if "PauliZ" in name:
            assert np.isclose(c, -h / 2)
        elif "PauliX" in name:
            assert np.isclose(c, -lam / (2 * n))
        elif "PauliY" in name:
            assert np.isclose(c, -gamma * lam / (2 * n))
        else:
            raise AssertionError(f"unexpected term {op}")


def test_hermitian():
    mat = dense_matrix(lmg_hamiltonian(4, h=1.0, lam=1.5, gamma=0.4), 4)
    assert np.allclose(mat, mat.conj().T)


@pytest.mark.parametrize("n", [2, 4, 6])
def test_parity_commutes(n):
    mat = dense_matrix(lmg_hamiltonian(n, h=1.0, lam=1.2, gamma=0.6), n)
    parity = dense_matrix(qml.Hamiltonian([1.0], [qml.prod(*(qml.PauliZ(i) for i in range(n)))]), n)
    assert np.linalg.norm(mat @ parity - parity @ mat) < 1e-12


@pytest.mark.parametrize("n", [2, 4, 6])
def test_total_spin_commutes(n):
    mat = dense_matrix(lmg_hamiltonian(n, h=1.0, lam=1.2, gamma=0.6), n)
    j2 = np.zeros_like(mat)
    for pauli in (qml.PauliX, qml.PauliY, qml.PauliZ):
        ja = 0.5 * sum(dense_matrix(qml.Hamiltonian([1.0], [pauli(i)]), n) for i in range(n))
        j2 = j2 + ja @ ja
    assert np.linalg.norm(mat @ j2 - j2 @ mat) < 1e-10


def test_heisenberg_matches_reference_structure():
    n = 4
    ham = heisenberg_xxz_hamiltonian(n, j1=10.0, j2=10.0)
    coeffs, ops = ham.terms()
    # 3 Pauli terms per bond + n Z-field terms
    assert len(coeffs) == 3 * (n - 1) + n
    assert all(np.isclose(c, 10.0) for c in coeffs)
    # Each bond term must be a simple Pauli product, not a Sum
    for op in ops[: 3 * (n - 1)]:
        assert not isinstance(op, qml.ops.Sum), f"got Sum instead of PauliProduct: {op}"


def test_heisenberg_pauli_rep_has_correct_term_count():
    """ham.pauli_rep must have 3*(N-1)+N distinct Pauli words, not 1 per bond."""
    n = 4
    ham = heisenberg_xxz_hamiltonian(n, j1=10.0, j2=10.0)
    pr = ham.pauli_rep
    assert pr is not None
    assert len(pr) == 3 * (n - 1) + n
