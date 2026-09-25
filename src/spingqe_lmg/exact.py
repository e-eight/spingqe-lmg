"""Exact diagonalization utilities.

Two routes, both funneled through ``ground_state(matrix)``:

- Dicke (``dicke_matrix``): the LMG Hamiltonian conserves J^2, and the ground
  state lives in the symmetric j = N/2 sector, an (N+1)-dimensional space.
  Diagonalizing there scales to N ~ 10^4 and is the validation ground truth
  for production sweeps.
- Dense (``dense_matrix``): brute-force 2^N diagonalization of any PennyLane
  Hamiltonian; cross-checks the Dicke route for small N.
"""

from collections.abc import Sequence

import numpy as np
import pennylane as qml

from spingqe_lmg.enums import HamiltonianKind
from spingqe_lmg.hamiltonians import heisenberg_xxz_hamiltonian


def collective_spin_matrices(n_qubits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(J_x, J_y, J_z) in the Dicke basis |j=N/2, m>, m = j, j-1, ..., -j."""
    j = n_qubits / 2
    m = np.arange(j, -j - 1, -1)
    jz = np.diag(m)
    # J+ |j, m> = sqrt(j(j+1) - m(m+1)) |j, m+1>; with rows ordered by descending m,
    # |j, m+1> is one row above |j, m>.
    raising = np.sqrt(j * (j + 1) - m[1:] * (m[1:] + 1))
    jp = np.diag(raising, k=1)
    jm = jp.T
    jx = (jp + jm) / 2
    jy = -0.5j * (jp - jm)  # (J+ - J-) / (2i)
    return jx, jy, jz


def dicke_matrix(n_qubits: int, h: float, lam: float, gamma: float = 0.0) -> np.ndarray:
    """LMG Hamiltonian restricted to the j = N/2 sector, shape (N+1, N+1).

    H = -h J_z - (lam/N)(J_x^2 + gamma J_y^2) + lam(1+gamma)/4, which equals the
    individual-spin H of hamiltonians.lmg_hamiltonian on this sector (the constant
    comes from sum_{i!=j} sigma_a sigma_a = 4 J_a^2 - N).
    """
    if n_qubits < 2:
        raise ValueError(f"n_qubits must be >= 2 for the LMG Hamiltonian (got {n_qubits})")
    jx, jy, jz = collective_spin_matrices(n_qubits)
    dim = n_qubits + 1
    ham = -h * jz - (lam / n_qubits) * (jx @ jx + gamma * (jy @ jy))
    ham = ham + lam * (1 + gamma) / 4 * np.eye(dim)
    return np.real_if_close(ham)


def dense_matrix(ham: qml.Hamiltonian, n_qubits: int) -> np.ndarray:
    """Dense 2^N matrix of a PennyLane Hamiltonian (N <= ~12)."""
    return qml.matrix(ham, wire_order=range(n_qubits))


def ground_state(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    """(E0, ground-state vector) via dense Hermitian diagonalization.

    Basis is whatever ``matrix`` is: pass ``dicke_matrix(...)`` for the Dicke
    sector or ``dense_matrix(...)`` for the full 2^N qubit basis.
    """
    vals, vecs = np.linalg.eigh(matrix)
    return float(vals[0]), vecs[:, 0]


def ground_energy(hc) -> float:
    """Exact ground-state energy for the given HamiltonianConfig.

    Routes to the Dicke sector for LMG (O(N) cost) and dense diagonalization
    for other Hamiltonians (O(2^N) cost, N <= ~12).
    """
    if hc.kind is HamiltonianKind.LMG:
        return ground_state(dicke_matrix(hc.n_qubits, hc.h, hc.lam, hc.gamma))[0]
    ham = heisenberg_xxz_hamiltonian(hc.n_qubits, hc.j1, hc.j2)
    return ground_state(dense_matrix(ham, hc.n_qubits))[0]


def exact_qpt_curve(
    n_qubits: int, h: float, lams: Sequence[float], gamma: float = 0.0
) -> np.ndarray:
    """Exact ground energies across a coupling sweep (Dicke route)."""
    return np.array([ground_state(dicke_matrix(n_qubits, h, lam, gamma))[0] for lam in lams])


def order_parameter_exact(n_qubits: int, h: float, lam: float, gamma: float = 0.0) -> float:
    """<(2 J_x / N)^2> in the exact ground state — the QPT order parameter."""
    if n_qubits < 2:
        raise ValueError(f"n_qubits must be >= 2 for the LMG Hamiltonian (got {n_qubits})")
    jx, _, _ = collective_spin_matrices(n_qubits)
    _, psi = ground_state(dicke_matrix(n_qubits, h, lam, gamma))
    val = psi.conj() @ (jx @ jx) @ psi
    return float(np.real(val) * 4 / n_qubits**2)
