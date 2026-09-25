"""PennyLane circuit-based energy evaluator.

Uses qml.device + qml.expval to evaluate sequence energies. Supports
shot-based (noisy) measurement; the only evaluator with this capability.
Called the "PennyLane evaluator" in the codebase; lightning.qubit is the
recommended device for CPU runs.
"""

from __future__ import annotations

import zlib
from functools import lru_cache

import numpy as np
import pennylane as qml

from spingqe_lmg.circuits import (
    prepare_initial_state,  # noqa: F401 — re-exported for backwards compat
)
from spingqe_lmg.core.caching import CachedEvaluatorMixin, LRUCache
from spingqe_lmg.pools import PoolOp
from spingqe_lmg.refine import RefineResult, refine_tokens


def _make_snapshot_qnode(device_name: str, n_qubits: int, shots: int = 0, seed: int | None = None):
    dev_kwargs = {"seed": seed} if seed is not None else {}
    dev = qml.device(device_name, wires=n_qubits, **dev_kwargs)

    @qml.qnode(dev)
    def circuit(ops, ham, init_angle):
        prepare_initial_state(n_qubits, init_angle)
        for op in ops:
            qml.Snapshot(measurement=qml.expval(ham))
            op.gate()
        return qml.expval(ham)

    qnode = circuit
    if shots:
        qnode = qml.set_shots(qnode, shots=shots)
    return qml.snapshots(qnode)


@lru_cache(maxsize=8)
def _analytic_snapshot_qnode(device_name: str, n_qubits: int):
    """Cached QNode factory — safe only because default.qubit treats Python-object
    Hamiltonians as dynamic hyperparameters. A JAX-autograph backend would freeze
    ``ham`` at the first call and silently return stale results for later ones."""
    return _make_snapshot_qnode(device_name, n_qubits)


def sequence_energies(
    ops: tuple[PoolOp, ...],
    ham: qml.Hamiltonian,
    n_qubits: int,
    device_name: str = "default.qubit",
    init_angle: float = 0.0,
    shots: int = 0,
    seed: int | None = None,
) -> np.ndarray:
    """Energies after each prefix of the gate sequence, shape (len(ops),).

    With shots > 0 a fresh seeded device is built per call: a cached device's
    RNG advances between executions, which would break same-seed
    reproducibility. Analytic evaluation keeps the cached QNode.
    """
    if len(ops) == 0:
        return np.zeros(0)
    if shots:
        qnode = _make_snapshot_qnode(device_name, n_qubits, shots, seed)
    else:
        qnode = _analytic_snapshot_qnode(device_name, n_qubits)
    snaps = qnode(ops, ham, init_angle)
    keys = list(range(1, len(ops))) + ["execution_results"]
    return np.array([float(np.real(snaps[k])) for k in keys])


def initial_energy(
    ham: qml.Hamiltonian,
    n_qubits: int,
    init_angle: float = 0.0,
    device_name: str = "default.qubit",
) -> float:
    """Energy of the reference state (the BOS-only / empty circuit).

    Computed as a device expectation value — never via ``qml.matrix(ham)``,
    whose dense 2^N x 2^N intermediates exceed available memory at N >= 16
    (the statevector is only 2^N elements).
    """
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev)
    def reference():
        prepare_initial_state(n_qubits, init_angle)
        return qml.expval(ham)

    return float(reference())


class PennyLaneEvaluator(CachedEvaluatorMixin):
    """Evaluate batches of pool-index sequences, with caching and parallelism.

    ``idx_batch`` rows are pool indices (token - 1, BOS stripped), shape (B, L).
    Returns per-step energies, shape (B, L).
    """

    def __init__(
        self,
        pool: list[PoolOp],
        ham: qml.Hamiltonian,
        n_qubits: int,
        device_name: str = "default.qubit",
        n_jobs: int = 1,
        init_angle: float = 0.0,
        shots: int = 0,
        seed: int | None = None,
    ):
        self.pool = list(pool)
        self.ham = ham
        self.n_qubits = n_qubits
        self.device_name = device_name
        self.n_jobs = n_jobs
        self.init_angle = init_angle
        self.shots = shots
        self.seed = seed
        # NOTE (shots > 0): the cache stores one noisy draw per unique token
        # sequence — repeats reuse that draw rather than re-sampling. This
        # is the measurement-frugal choice (matches how recorded lab data
        # would be reused).
        self._cache: LRUCache = LRUCache(maxsize=None)

    def refine_tokens(
        self,
        tokens: list[int],
        *,
        qml_device: str = "default.qubit",
        shots: int = 0,
        seed: int = 0,
        optimizer=None,
        gradient=None,
    ) -> RefineResult:
        return refine_tokens(
            tokens,
            self.pool,
            self.ham,
            self.n_qubits,
            device_name=qml_device,
            init_angle=self.init_angle,
            shots=shots,
            seed=seed,
            optimizer=optimizer,
            gradient=gradient,
        )

    def initial_energy(self) -> float:
        """Energy of the reference state (the BOS-only / empty circuit)."""
        return initial_energy(self.ham, self.n_qubits, self.init_angle, self.device_name)

    def _ops_for(self, key: tuple[int, ...]) -> tuple[PoolOp, ...]:
        return tuple(self.pool[i] for i in key)

    def _sequence_energies(self, key: tuple[int, ...]) -> np.ndarray:
        return sequence_energies(
            self._ops_for(key),
            self.ham,
            self.n_qubits,
            self.device_name,
            self.init_angle,
            self.shots,
            self._seed_for(key),
        )

    def _seed_for(self, key: tuple[int, ...]) -> int | None:
        """Per-sequence seed: reproducible AND independent across sequences."""
        if not self.shots:
            return None
        return zlib.crc32(repr((self.seed, key)).encode())


class PennyLaneGroupedEvaluator(PennyLaneEvaluator):
    r"""PennyLane evaluator using grouped commuting observables.

    The standard ``PennyLaneEvaluator`` passes the full Hamiltonian to
    ``qml.expval``, which ``lightning.qubit`` evaluates term-by-term,
    materialising :math:`\mathcal{O}(N^2)` statevector copies per
    Snapshot — the source of the :math:`\sim 600\times` RSS overhead
    reported in the SC manuscript.

    This variant partitions the Hamiltonian into qubit-wise commuting
    groups via :func:`qml.pauli.group_observables` and measures each
    group in a single basis-rotation pass, collapsing the per-call
    memory footprint.

    .. warning::

       This evaluator computes *only the final energy* and broadcasts
       it to all ``len(key)`` steps.  It is suitable for benchmarking
       throughput but NOT for GQE training, which requires per-step
       energy curves.  Use ``PennyLaneEvaluator`` for training.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._groups = qml.pauli.group_observables(self.ham)

    def _sequence_energies(self, key: tuple[int, ...]) -> np.ndarray:
        if len(key) == 0:
            return np.zeros(0)

        ops = self._ops_for(key)
        dev = qml.device(self.device_name, wires=self.n_qubits)

        @qml.qnode(dev)
        def _circuit():
            prepare_initial_state(self.n_qubits, self.init_angle)
            for op in ops:
                op.gate()
            return [
                qml.expval(qml.sum(*group))
                for group in self._groups
            ]

        results = _circuit()
        total = sum(float(r) for r in results)
        return np.full(len(key), total)
