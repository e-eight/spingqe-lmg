"""Angle refinement of generated sequences.

Adapted from ``postprocessing.ipynb`` in Mindbeam-AI/SpinGQE (MIT): the GQE
training loop discovers good circuit *structures*, but its operator pool only
contains a fixed discrete set of angles, so the raw minimum sampled energy
plateaus above the ground state. Refinement keeps the gate structure of the
best sequence and locally optimizes all rotation angles.

Works for both pool kinds (each PoolOp carries exactly one angle).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pennylane as qml
from scipy.optimize import minimize

from spingqe_lmg.circuits import build_energy_qnode
from spingqe_lmg.enums import RefineGradient, RefineOptimizer
from spingqe_lmg.pools import PoolOp


@dataclass
class RefineResult:
    energy: float
    angles: list[float]
    initial_energy: float
    n_evaluations: int
    n_gradient_evaluations: int = 0


def refine_angles(
    ops: Sequence[PoolOp],
    ham: qml.Hamiltonian,
    n_qubits: int,
    device_name: str = "default.qubit",
    init_angle: float = 0.0,
    x0: Sequence[float] | None = None,
    gradient: RefineGradient = RefineGradient.ADJOINT,
) -> RefineResult:
    """Locally optimize the angles of a fixed gate sequence.

    ``x0`` overrides the starting angles (default: each op's pool angle).
    ``gradient``: "adjoint" (analytic jacobian) or "finite_diff" (scipy's
    default finite-difference behavior, kept as an escape hatch).
    """
    if gradient not in (RefineGradient.ADJOINT, RefineGradient.FINITE_DIFF):
        raise ValueError(f"unknown gradient method: {gradient!r}")

    if gradient is RefineGradient.ADJOINT:
        cost = build_energy_qnode(
            ops, ham, n_qubits, init_angle, device_name, diff_method="adjoint"
        )
        grad_fn = qml.grad(cost)

        def fun(p):
            return float(cost(qml.numpy.array(p, requires_grad=True)))

        def jac(p):
            return np.asarray(grad_fn(qml.numpy.array(p, requires_grad=True)), dtype=float)
    else:
        cost = build_energy_qnode(ops, ham, n_qubits, init_angle, device_name)

        def fun(p):
            return float(cost(p))

        jac = None

    x0 = np.array([op.angle for op in ops] if x0 is None else x0, dtype=float)
    initial = fun(x0)
    res = minimize(fun, x0, jac=jac, method="L-BFGS-B", tol=1e-10)
    return RefineResult(
        energy=float(res.fun),
        angles=[float(a) for a in res.x],
        initial_energy=initial,
        n_evaluations=int(res.nfev),
        n_gradient_evaluations=int(getattr(res, "njev", 0)) if jac is not None else 0,
    )


def refine_angles_spsa(
    ops: Sequence[PoolOp],
    ham: qml.Hamiltonian,
    n_qubits: int,
    shots: int,
    seed: int = 0,
    iterations: int = 300,
    device_name: str = "default.qubit",
    init_angle: float = 0.0,
    a: float = 0.15,
    c: float = 0.1,
    stability_constant: float = 0.0,
    x0: np.ndarray | None = None,
) -> RefineResult:
    """SPSA angle refinement under finite shots.

    Standard SPSA gains a_k = a/(k+1+A)^0.602, c_k = c/(k+1)^0.101; two
    shot-based cost evaluations per iteration.  The returned ``energy`` is the
    *analytic* energy of the final angles.  Set ``stability_constant`` (Spall's
    *A*) to ~10% of ``iterations`` to damp early overshoot (default 0.0 for
    backward compatibility)."""
    noisy_cost = build_energy_qnode(
        ops, ham, n_qubits, init_angle, device_name, shots=shots, seed=seed
    )

    rng = np.random.default_rng(seed)
    if x0 is not None:
        x = np.asarray(x0, dtype=float)
    else:
        x = np.array([op.angle for op in ops], dtype=float)
    initial = float(noisy_cost(x))
    n_evals = 1
    for k in range(iterations):
        ak = a / (k + 1 + stability_constant) ** 0.602
        ck = c / (k + 1) ** 0.101
        delta = rng.choice([-1.0, 1.0], size=len(x))
        g = (
            (float(noisy_cost(x + ck * delta)) - float(noisy_cost(x - ck * delta)))
            / (2 * ck)
            * delta
        )
        n_evals += 2
        x = x - ak * g

    exact_cost = build_energy_qnode(ops, ham, n_qubits, init_angle, device_name)

    return RefineResult(
        energy=float(exact_cost(x)),
        angles=[float(v) for v in x],
        initial_energy=initial,
        n_evaluations=n_evals,
    )


def refine_angles_cobyla(
    ops: Sequence[PoolOp],
    ham: qml.Hamiltonian,
    n_qubits: int,
    shots: int,
    seed: int = 0,
    max_evaluations: int = 601,
    device_name: str = "default.qubit",
    init_angle: float = 0.0,
) -> RefineResult:
    """COBYLA refinement under finite shots, for optimizer comparison.

    COBYLA's interpolation model assumes consistent evaluations and degrades
    under sampling noise relative to SPSA. Evaluation budget matched to SPSA's
    (1 + 2*iterations). Returns the analytic energy of the final angles.
    """
    noisy_cost = build_energy_qnode(
        ops, ham, n_qubits, init_angle, device_name, shots=shots, seed=seed
    )

    x0 = np.array([op.angle for op in ops], dtype=float)
    initial = float(noisy_cost(x0))
    res = minimize(
        lambda p: float(noisy_cost(p)),
        x0,
        method="COBYLA",
        options={"maxiter": max_evaluations - 1, "rhobeg": 0.3},
    )

    exact_cost = build_energy_qnode(ops, ham, n_qubits, init_angle, device_name)

    return RefineResult(
        energy=float(exact_cost(res.x)),
        angles=[float(v) for v in res.x],
        initial_energy=initial,
        n_evaluations=int(res.nfev) + 1,
    )


def refine_tokens(
    tokens: Sequence[int],
    pool: list[PoolOp],
    ham: qml.Hamiltonian,
    n_qubits: int,
    device_name: str = "default.qubit",
    init_angle: float = 0.0,
    shots: int = 0,
    seed: int = 0,
    optimizer: RefineOptimizer = RefineOptimizer.SPSA,
    gradient: RefineGradient = RefineGradient.ADJOINT,
) -> RefineResult:
    """Refine a token sequence (BOS at position 0, as stored in best_sequence.json).

    With shots > 0 the optimization runs under sampled energies (SPSA default;
    COBYLA for literature comparison); otherwise analytic L-BFGS, with
    ``gradient`` selecting adjoint vs finite-difference derivatives.
    """
    if any(t == 0 for t in tokens[1:]):
        raise ValueError(f"BOS token at non-zero position in tokens: {tokens}")
    ops = [pool[t - 1] for t in tokens[1:]]
    if shots:
        if optimizer is RefineOptimizer.COBYLA:
            return refine_angles_cobyla(
                ops,
                ham,
                n_qubits,
                shots,
                seed=seed,
                device_name=device_name,
                init_angle=init_angle,
            )
        if optimizer is not RefineOptimizer.SPSA:
            raise ValueError(f"unknown shot-based optimizer: {optimizer!r}")
        return refine_angles_spsa(
            ops, ham, n_qubits, shots, seed=seed, device_name=device_name, init_angle=init_angle
        )
    return refine_angles(
        ops, ham, n_qubits, device_name=device_name, init_angle=init_angle, gradient=gradient
    )
