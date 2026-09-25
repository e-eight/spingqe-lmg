"""CUDA-Q energy-evaluation backend.

Drop-in alternative to the PennyLane snapshot evaluator, selected with
``train.evaluator = "cudaq"``. Uses CUDA-Q's GPU-accelerated statevector
simulation (cuStateVec), which dominates CPU statevector paths above
approximately 20 qubits.

Implementation:
- Every pool gate reduces to PauliRot factors via ``PoolOp.factors()``; the
  kernel applies them with ``exp_pauli`` (exp(i c P) convention, c = -theta/2).
- The Hamiltonian converts through PennyLane's ``pauli_rep`` into a
  ``cudaq.SpinOperator``.
- Default mode (v2): chains state handles — after each gate the statevector is
  captured with ``cudaq.get_state`` and the next gate resumes from it, so a
  length-L sequence costs O(L) gate applications. The original prefix-circuit
  path (v1, O(L^2/2)) is kept behind ``chain_states=False`` as an equivalence
  reference.
- Target: "nvidia" when a GPU is visible, else "qpp-cpu".
"""

import numpy as np

from spingqe_lmg.core.caching import CachedEvaluatorMixin, LRUCache
from spingqe_lmg.pools import PoolOp
from spingqe_lmg.refine import RefineResult, refine_tokens

try:
    import cudaq

    CUDAQ_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra dep
    CUDAQ_AVAILABLE = False


def _set_target() -> str:
    if cudaq.num_available_gpus() > 0 and cudaq.has_target("nvidia"):
        target = "nvidia"
    else:
        target = "qpp-cpu"
    cudaq.set_target(target)
    return target


def spin_op_from_pennylane(ham, n_qubits: int):
    """Convert a PennyLane Hamiltonian to a cudaq.SpinOperator via pauli_rep."""
    sentence = ham.pauli_rep
    if sentence is None:
        raise ValueError("Hamiltonian has no Pauli representation")
    op = 0.0 * cudaq.spin.i(0)
    table = {"X": cudaq.spin.x, "Y": cudaq.spin.y, "Z": cudaq.spin.z}
    for word, coeff in sentence.items():
        term = float(np.real(coeff)) * cudaq.spin.i(n_qubits - 1)
        for wire, pauli in word.items():
            term = term * table[pauli](int(wire))
        op += term
    return op


def _full_word(word: str, wires: tuple[int, ...], n_qubits: int) -> str:
    out = ["I"] * n_qubits
    for pauli, wire in zip(word, wires, strict=True):
        out[wire] = pauli
    return "".join(out)


if CUDAQ_AVAILABLE:

    @cudaq.kernel
    def _sequence_kernel(
        n_qubits: int,
        init_angle: float,
        words: list[cudaq.pauli_word],
        coefficients: list[float],
    ):
        q = cudaq.qvector(n_qubits)
        if init_angle != 0.0:
            for i in range(n_qubits):
                ry(init_angle, q[i])  # noqa: F821 - cudaq kernel builtin
        for k in range(len(words)):
            exp_pauli(coefficients[k], q, words[k])  # noqa: F821 - cudaq kernel builtin

    @cudaq.kernel
    def _init_kernel(n_qubits: int, init_angle: float):
        q = cudaq.qvector(n_qubits)
        if init_angle != 0.0:
            for i in range(n_qubits):
                ry(init_angle, q[i])  # noqa: F821 - cudaq kernel builtin

    @cudaq.kernel
    def _step_kernel(state: cudaq.State, words: list[cudaq.pauli_word], coefficients: list[float]):
        q = cudaq.qvector(state)
        for k in range(len(words)):
            exp_pauli(coefficients[k], q, words[k])  # noqa: F821 - cudaq kernel builtin

    @cudaq.kernel
    def _hold_kernel(state: cudaq.State):
        q = cudaq.qvector(state)  # noqa: F841 - identity kernel: observe on a held state


class CudaQEvaluator(CachedEvaluatorMixin):
    """CUDA-Q evaluator using the NVIDIA CUDA-Q runtime for circuit simulation.

    Same contract as PennyLaneEvaluator: ``idx_batch`` rows are pool
    indices (token - 1), returns per-step energies of shape (B, L).
    """

    def __init__(
        self,
        pool: list[PoolOp],
        ham,
        n_qubits: int,
        init_angle: float = 0.0,
        n_jobs: int = 1,  # accepted for interface parity; batching is internal
        chain_states: bool = True,  # False: v1 one-prefix-circuit-per-step path
        cache_maxsize: int | None = None,
    ):
        if not CUDAQ_AVAILABLE:
            raise ImportError("cudaq is not installed; pip install cudaq")
        self.pool = list(pool)
        self.n_qubits = n_qubits
        if isinstance(init_angle, str):
            raise NotImplementedError(
                "the cudaq evaluator supports only product-state (RY angle) references"
            )
        self.init_angle = float(init_angle)
        self.chain_states = chain_states
        self.target = _set_target()
        self.ham = ham
        self.spin_op = spin_op_from_pennylane(ham, n_qubits)
        self._cache: LRUCache = LRUCache(cache_maxsize)
        # pre-expand each pool op into (full-register word, exp_pauli coefficient)
        self._op_factors: list[list[tuple[str, float]]] = [
            [(_full_word(word, wires, n_qubits), -theta / 2) for word, wires, theta in op.factors()]
            for op in self.pool
        ]

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
        """Energy of the reference state <psi_0| H |psi_0>, computed via the
        evaluator's own CUDA-Q path."""
        return self._prefix_energy([])

    def _prefix_energy(self, factors: list[tuple[str, float]]) -> float:
        words = [cudaq.pauli_word(w) for w, _ in factors]
        coefficients = [c for _, c in factors]
        result = cudaq.observe(
            _sequence_kernel, self.spin_op, self.n_qubits, self.init_angle, words, coefficients
        )
        return float(result.expectation())

    def _sequence_energies_prefix(self, key: tuple[int, ...]) -> np.ndarray:
        """v1: one prefix circuit per step — O(L^2/2) gate applications."""
        per_gate = [self._op_factors[i] for i in key]
        energies = []
        flat: list[tuple[str, float]] = []
        for gate_factors in per_gate:
            flat = flat + gate_factors
            energies.append(self._prefix_energy(flat))
        return np.array(energies)

    def _sequence_energies(self, key: tuple[int, ...]) -> np.ndarray:
        """v2: chain state handles — O(L) gate applications per sequence."""
        if not self.chain_states:
            return self._sequence_energies_prefix(key)
        state = cudaq.get_state(_init_kernel, self.n_qubits, self.init_angle)
        energies = []
        for i in key:
            factors = self._op_factors[i]
            words = [cudaq.pauli_word(w) for w, _ in factors]
            coefficients = [c for _, c in factors]
            state = cudaq.get_state(_step_kernel, state, words, coefficients)
            result = cudaq.observe(_hold_kernel, self.spin_op, state)
            energies.append(float(result.expectation()))
        return np.array(energies)
