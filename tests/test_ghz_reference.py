"""Tests for the parity-even GHZ-x stabilizer reference (init_state="ghz_x").

The state is (|+...+> + |-...->)/sqrt(2) — the lowest-energy entangled
stabilizer state of the LMG Hamiltonian (Robin, PRA 112, 052408). For our
convention H = -(h/2) sum Z - (lam/4N) sum_{i!=j} (XX + gamma YY) it has
<Z_i> = 0, <X_i X_j> = 1 and <Y_i Y_j> = 0 (N > 2), so E = -lam (N-1)/4, and
at h = 0 it is the exact ground state.
"""

import numpy as np
import pennylane as qml
import pytest

from spingqe_lmg.config import config_from_dict
from spingqe_lmg.evaluators.dicke import DickeSpace
from spingqe_lmg.evaluators.pennylane import (
    initial_energy,
    prepare_initial_state,
    sequence_energies,
)
from spingqe_lmg.exact import dicke_matrix, ground_state
from spingqe_lmg.hamiltonians import lmg_hamiltonian
from spingqe_lmg.pools import build_pauli_pair_pool
from spingqe_lmg.training import initial_angle_for


def _ghz_x_statevector(n: int) -> np.ndarray:
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit():
        prepare_initial_state(n, "ghz_x")
        return qml.state()

    return np.asarray(circuit())


@pytest.mark.parametrize("n", [2, 4, 6, 8])
def test_ghz_x_statevector_is_even_parity_x_basis_ghz(n):
    psi = _ghz_x_statevector(n)
    signs = np.array([(-1) ** bin(z).count("1") for z in range(2**n)])
    plus = np.full(2**n, 2 ** (-n / 2))
    target = (plus + signs * plus) / np.sqrt(2)
    assert np.abs(psi.conj() @ target) ** 2 == pytest.approx(1.0, abs=1e-12)
    # parity <prod Z> = +1
    assert np.sum(signs * np.abs(psi) ** 2) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("n", [4, 8])
@pytest.mark.parametrize("gamma", [0.0, 0.5])
def test_ghz_x_initial_energy_is_stabilizer_value(n, gamma):
    h, lam = 1.0, 1.5
    ham = lmg_hamiltonian(n, h=h, lam=lam, gamma=gamma)
    assert initial_energy(ham, n, "ghz_x") == pytest.approx(-lam * (n - 1) / 4, abs=1e-10)


def test_ghz_x_is_exact_ground_state_at_h_zero():
    n, lam = 8, 1.5
    ham = lmg_hamiltonian(n, h=0.0, lam=lam)
    assert initial_energy(ham, n, "ghz_x") == pytest.approx(
        ground_state(dicke_matrix(n, 0.0, lam))[0], abs=1e-10
    )


def test_sequence_energies_respect_ghz_reference():
    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=1.5)
    pool = build_pauli_pair_pool(n, angles=(1e-12,), paulis=("ZZ",), include_single_z=False)
    es = sequence_energies((pool[0],), ham, n, init_angle="ghz_x")
    assert es[0] == pytest.approx(initial_energy(ham, n, "ghz_x"), abs=1e-9)


def test_ghz_x_rejects_odd_n():
    ham = lmg_hamiltonian(3, h=1.0, lam=1.0)
    with pytest.raises(ValueError, match="even n_qubits"):
        initial_energy(ham, 3, "ghz_x")


def test_unknown_string_reference_rejected():
    ham = lmg_hamiltonian(4, h=1.0, lam=1.0)
    with pytest.raises(ValueError, match="unknown reference state"):
        initial_energy(ham, 4, "ghz_y")


def test_initial_angle_for_ghz_config():
    cfg = config_from_dict(
        {"hamiltonian": {"kind": "lmg", "n_qubits": 4}, "train": {"init_state": "ghz_x"}}
    )
    assert initial_angle_for(cfg) == "ghz_x"


def test_dicke_evaluator_rejects_ghz_reference():
    space = DickeSpace.build(4, h=1.0, lam=1.5, gamma=0.0)
    with pytest.raises(NotImplementedError, match="product-state"):
        space.initial_state("ghz_x")


@pytest.mark.parametrize("n", [2, 4, 6, 8])
def test_ghz_x_cross_backend_consistency(n):
    """Circuit (energy.py) and numpy (core/statevec_ops.py) GHZ-X statevectors match."""
    from spingqe_lmg.core.statevec_ops import initial_statevector

    psi_circuit = _ghz_x_statevector(n)
    psi_numpy = initial_statevector(n, "ghz_x")
    assert np.abs(psi_circuit.conj() @ psi_numpy) ** 2 == pytest.approx(1.0, abs=1e-12)
