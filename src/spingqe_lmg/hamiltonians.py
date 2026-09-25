"""Hamiltonian builders.

LMG conventions (individual spin-1/2 encoding):

    H = -(h/2) sum_i Z_i - (lam / 4N) [ sum_{i!=j} X_i X_j + gamma * sum_{i!=j} Y_i Y_j ]

Since sum_{i!=j} = 2 * sum_{i<j}, the per-pair coefficients used below are
-lam/(2N) for X_i X_j and -gamma*lam/(2N) for Y_i Y_j. In the j = N/2 Dicke
sector this is H = -h J_z - (lam/N)(J_x^2 + gamma J_y^2) + lam(1+gamma)/4
(see exact.dicke_matrix). The quantum phase transition sits at lam_c = h.

All functions in this module return total <H> (extensive in N, not per-spin).
To compare across system sizes, divide by N.
"""

import math

import pennylane as qml

from spingqe_lmg.enums import HamiltonianKind


def lmg_hamiltonian(
    n_qubits: int, h: float = 1.0, lam: float = 1.0, gamma: float = 0.0
) -> qml.Hamiltonian:
    """LMG Hamiltonian on ``n_qubits`` spins with individual spin encoding.

    Y_i Y_j terms are included only when gamma != 0.
    """
    if n_qubits < 2:
        raise ValueError("LMG model needs at least 2 spins")
    coeffs = []
    ops = []
    for i in range(n_qubits):
        coeffs.append(-h / 2)
        ops.append(qml.PauliZ(i))
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            coeffs.append(-lam / (2 * n_qubits))
            ops.append(qml.PauliX(i) @ qml.PauliX(j))
            if gamma != 0.0:
                coeffs.append(-gamma * lam / (2 * n_qubits))
                ops.append(qml.PauliY(i) @ qml.PauliY(j))
    return qml.Hamiltonian(coeffs, ops)


def heisenberg_xxz_hamiltonian(n_qubits: int, j1: float = 1.0, j2: float = 2.0) -> qml.Hamiltonian:
    """Nearest-neighbor Heisenberg chain with a Z field.

    Faithful port of ``hamiltonian.one`` from the SpinGQE reference
    (github.com/Mindbeam-AI/SpinGQE, MIT); used for the port-validation baseline.
    """
    coeffs = []
    ops = []
    for i in range(n_qubits - 1):
        for P in (qml.PauliX, qml.PauliY, qml.PauliZ):
            ops.append(P(i) @ P(i + 1))
            coeffs.append(j1)
    for i in range(n_qubits):
        coeffs.append(j2)
        ops.append(qml.PauliZ(i))
    return qml.Hamiltonian(coeffs, ops)


def lmg_mean_field_angle(h: float, lam: float) -> float:
    """Tilt angle of the broken-symmetry mean-field product state (gamma = 0).

    Minimizing E(theta) = -(Nh/2) cos(theta) - (lam (N-1)/4) sin^2(theta) over
    product states tilted by theta in the x-z plane gives cos(theta) = h/lam
    for lam > h and theta = 0 otherwise (large-N mean field; we use it as a
    reference state, so the finite-N correction is irrelevant).
    """
    if lam <= h or lam == 0.0:
        return 0.0
    return math.acos(h / lam)


def build_hamiltonian(cfg) -> qml.Hamiltonian:
    """Dispatch on a HamiltonianConfig."""
    if cfg.kind is HamiltonianKind.LMG:
        return lmg_hamiltonian(cfg.n_qubits, h=cfg.h, lam=cfg.lam, gamma=cfg.gamma)
    if cfg.kind is HamiltonianKind.HEISENBERG_XXZ:
        return heisenberg_xxz_hamiltonian(cfg.n_qubits, j1=cfg.j1, j2=cfg.j2)
    raise ValueError(f"unknown hamiltonian kind: {cfg.kind!r}")
