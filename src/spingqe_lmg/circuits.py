"""Shared circuit builders for refinement, ADAPT-VQE, and state-quality analysis.

Avoids duplicating the ``prepare_initial_state -> apply gates -> expval(ham)``
pattern across multiple modules.
"""

from collections.abc import Sequence

import numpy as np
import pennylane as qml

from spingqe_lmg.pools import PoolOp


def prepare_initial_state(n_qubits: int, init_angle: float | str) -> None:
    """Queue the reference-state preparation.

    A float is the RY tilt angle on every qubit: 0 is |0...0>, theta_mf the
    broken-symmetry mean-field product state (hamiltonians.lmg_mean_field_angle).
    The string "ghz_x" prepares the parity-even GHZ state in the X basis,
    (|+...+> + |-...->)/sqrt(2) — the lowest-energy entangled stabilizer state
    of the LMG Hamiltonian (Robin, PRA 112, 052408). Defined for even n_qubits
    (the even-parity sector, where the even-N LMG ground state lives).
    """
    if isinstance(init_angle, str):
        if init_angle != "ghz_x":
            raise ValueError(f"unknown reference state: {init_angle!r}")
        if n_qubits % 2:
            raise ValueError("init_state='ghz_x' is defined for even n_qubits")
        qml.Hadamard(wires=0)
        for w in range(1, n_qubits):
            qml.CNOT(wires=[0, w])
        for w in range(n_qubits):
            qml.Hadamard(wires=w)
    elif init_angle == 0.0:
        qml.BasisState(np.zeros(n_qubits, dtype=int), wires=range(n_qubits))
    else:
        for w in range(n_qubits):
            qml.RY(init_angle, wires=w)


def build_energy_qnode(
    ops: Sequence[PoolOp],
    ham: qml.Hamiltonian,
    n_qubits: int,
    init_angle: float,
    device_name: str = "default.qubit",
    shots: int = 0,
    seed: int | None = None,
    diff_method: str | None = None,
) -> qml.QNode:
    """Return a QNode that computes ``<psi(theta)| H |psi(theta)>``.

    ``ops`` is the ordered list of pool operators (their angles are unified
    into a single parameter array).  ``shots > 0`` applies ``qml.set_shots``
    for finite-shot sampling (seed passed to the device for RNG reproducibility).
    ``diff_method`` selects the differentiation strategy
    (e.g. ``"adjoint"`` for analytic gradients).
    """
    dev_kwargs = {"seed": seed} if seed is not None else {}
    dev = qml.device(device_name, wires=n_qubits, **dev_kwargs)
    qnode_kwargs = {"diff_method": diff_method} if diff_method else {}

    @qml.qnode(dev, **qnode_kwargs)
    def circuit(params):
        prepare_initial_state(n_qubits, init_angle)
        for op, theta in zip(ops, params, strict=True):
            op.gate(angle=theta)
        return qml.expval(ham)

    if shots:
        circuit = qml.set_shots(circuit, shots=shots)
    return circuit
