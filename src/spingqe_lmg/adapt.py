"""Vanilla qubit-ADAPT-VQE baseline with fluctuation-derived stall repair.

**qubit-ADAPT** (Tang et al., arXiv:1911.10205; LMG prior art, PRC 105,
064317) — qubit-operator pool, not the original fermionic-excitation ADAPT
(Grimsley et al. 2019). At each iteration, screen every pool generator A by
the energy gradient of appending exp(-i theta/2 A) at theta = 0, append the
largest-|gradient| generator, then re-optimize all angles (L-BFGS via
refine_angles). Pauli-word candidates (P^2 = I) use the exact two-term
parameter-shift rule. Stop on convergence, max iterations, or a **stall**
(all screening gradients below tolerance).

**Stall mechanism**: the screening gradient is the first-order coupling — a
pool whose generators have no first-order coupling at the reference state
stalls by construction.

**Stall repair**: when stalled, derive the missing operator content from the
fluctuation theory of the target phase. For the LMG broken phase, the
Holstein-Primakoff squeezing generators yield two-qubit words YZ/ZY/XY/YX.
Applies to product-like references (mean-field, |0...0>).
"""

import inspect
from dataclasses import dataclass

import numpy as np
import pennylane as qml

from spingqe_lmg.circuits import prepare_initial_state
from spingqe_lmg.enums import Connectivity, PoolKind
from spingqe_lmg.pools import PoolOp
from spingqe_lmg.refine import refine_angles


def build_adapt_pool(
    n_qubits: int,
    paulis: tuple[str, ...] = ("ZZ", "XX", "YY"),
    connectivity: Connectivity = Connectivity.ALL,
    include_single_z: bool = True,
    kind: PoolKind = PoolKind.PAULI_PAIR,
    generators: tuple[str, ...] = ("Jz", "Jx2", "Jy2"),
) -> list[PoolOp]:
    """Unique-generator pool (one entry per generator; ADAPT owns the angles)."""
    if isinstance(kind, str):
        kind = PoolKind(kind)
    if isinstance(connectivity, str):
        connectivity = Connectivity(connectivity)
    if kind is PoolKind.COLLECTIVE:
        return [
            PoolOp(label=g, kind="collective", angle=0.0, n_qubits=n_qubits, generator=g)
            for g in generators
        ]
    if connectivity is Connectivity.ALL:
        pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    elif connectivity is Connectivity.NN:
        pairs = [(i, i + 1) for i in range(n_qubits - 1)]
    else:
        raise ValueError(f"unknown connectivity: {connectivity!r}")
    pool = [
        PoolOp(
            label=f"{word}({i},{j})",
            kind="pauli_rot",
            angle=0.0,
            n_qubits=n_qubits,
            pauli_word=word,
            wires=(i, j),
        )
        for word in paulis
        for (i, j) in pairs
    ]
    if include_single_z:
        pool += [
            PoolOp(
                label=f"Z({i})",
                kind="pauli_rot",
                angle=0.0,
                n_qubits=n_qubits,
                pauli_word="Z",
                wires=(i,),
            )
            for i in range(n_qubits)
        ]
    return pool


def lmg_repair_words(h: float, lam: float) -> tuple[str, ...]:
    """Pauli words of the HP squeezing generators for the LMG phase at (h, lam).

    Broken phase (lam > h): the quadratic fluctuation generators transverse to
    the tilted mean-field axis are {Jy, Jz} and {Jx, Jy} → YZ/ZY and XY/YX.
    Symmetric phase: the reference |0...0> already has the right structure at
    first order; no repair words are defined (returns empty).
    """
    if lam > h:
        return ("YZ", "ZY", "XY", "YX")
    return ()


@dataclass
class AdaptResult:
    energies: list[float]  # energy after each iteration's re-optimization
    gradients: list[float]  # max |screening gradient| at each iteration
    labels: list[str]  # selected generator per iteration
    angles: list[float]  # final optimized angles
    stalled: bool
    repaired_at: int | None  # iteration index where the pool was repaired
    n_circuit_evals: int
    initial_energy: float  # reference-state energy (always populated)
    final_energy: float | None  # converged energy (None if zero iterations)


class AdaptVQE:
    """Vanilla qubit-ADAPT-VQE with optional fluctuation-derived stall repair."""

    def __init__(
        self,
        pool: list[PoolOp],
        ham: qml.Hamiltonian,
        n_qubits: int,
        init_angle: float = 0.0,
        *,
        device_name: str = "default.qubit",
        grad_tol: float = 1e-4,
        grad_delta: float = 1e-3,
        h: float = 1.0,
        lam: float = 1.0,
    ):
        self.pool = list(pool)
        self.ham = ham
        self.n_qubits = n_qubits
        self.init_angle = init_angle
        self.device_name = device_name
        self.grad_tol = grad_tol
        self.grad_delta = grad_delta
        self.h = h
        self.lam = lam
        self.n_circuit_evals = 0
        dev = qml.device(device_name, wires=n_qubits)

        @qml.qnode(dev)
        def _energy(ops, params):
            prepare_initial_state(n_qubits, init_angle)
            for op, theta in zip(ops, params, strict=True):
                op.gate(angle=float(theta))
            return qml.expval(ham)

        self._energy = _energy

    def energy(self, ops: list[PoolOp], params: list[float]) -> float:
        self.n_circuit_evals += 1
        return float(self._energy(ops, params))

    def screening_gradients(self, ops: list[PoolOp], params: list[float]) -> np.ndarray:
        """|dE/dtheta| at theta = 0 for each candidate generator appended.

        Pauli-word candidates (P^2 = 1) use the exact two-term parameter-shift
        rule, [E(+pi/2) - E(-pi/2)]/2 — the same two evaluations as a central
        finite difference but with zero truncation error (and the
        hardware-native gradient). Collective candidates have multi-frequency
        generators, where the two-term rule is invalid; they keep the central
        finite difference with step ``grad_delta``.
        """
        grads = np.empty(len(self.pool))
        for k, cand in enumerate(self.pool):
            if cand.kind == "pauli_rot":
                shift = np.pi / 2
                e_plus = self.energy(ops + [cand], params + [shift])
                e_minus = self.energy(ops + [cand], params + [-shift])
                grads[k] = abs(e_plus - e_minus) / 2
            else:
                d = self.grad_delta
                e_plus = self.energy(ops + [cand], params + [d])
                e_minus = self.energy(ops + [cand], params + [-d])
                grads[k] = abs(e_plus - e_minus) / (2 * d)
        return grads

    def run(
        self,
        max_iterations: int = 30,
        target_error: float | None = None,
        exact_energy: float | None = None,
        repair: bool = False,
        repair_words_fn=None,
    ) -> AdaptResult:
        ops: list[PoolOp] = []
        params: list[float] = []
        energies, gradients, labels = [], [], []
        stalled = False
        repaired_at: int | None = None
        ref_energy = self.energy(ops, params)

        for iteration in range(max_iterations):
            grads = self.screening_gradients(ops, params)
            gmax = float(grads.max())
            gradients.append(gmax)

            if gmax < self.grad_tol:
                # stall: no first-order direction left in the pool
                if repair and repaired_at is None and repair_words_fn is not None:
                    sig = inspect.signature(repair_words_fn)
                    if len(sig.parameters) >= 2:
                        new_words = repair_words_fn(self.h, self.lam)
                    else:
                        new_words = repair_words_fn()
                    if new_words:
                        extra = build_adapt_pool(
                            self.n_qubits, paulis=tuple(new_words), include_single_z=False
                        )
                        self.pool = self.pool + extra
                        repaired_at = iteration
                        grads = self.screening_gradients(ops, params)
                        gmax = float(grads.max())
                        gradients[-1] = gmax
                if gmax < self.grad_tol:
                    stalled = True
                    break

            chosen = int(np.argmax(grads))
            ops.append(self.pool[chosen])
            labels.append(self.pool[chosen].label)
            result = refine_angles(
                ops,
                self.ham,
                self.n_qubits,
                device_name=self.device_name,
                init_angle=self.init_angle,
                x0=params + [0.0],
            )
            self.n_circuit_evals += result.n_evaluations
            params = list(result.angles)
            energies.append(result.energy)

            if (
                target_error is not None
                and exact_energy is not None
                and abs(result.energy - exact_energy) < target_error
            ):
                break

        final = energies[-1] if energies else None
        return AdaptResult(
            energies=energies,
            gradients=gradients,
            labels=labels,
            angles=params,
            stalled=stalled,
            repaired_at=repaired_at,
            n_circuit_evals=self.n_circuit_evals,
            initial_energy=ref_energy,
            final_energy=final,
        )
