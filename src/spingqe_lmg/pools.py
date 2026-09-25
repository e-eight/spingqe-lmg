"""Operator pools for GQE sequence generation.

Three variants, all parity-conserving (every op commutes with the spin-flip
parity P = Z x Z x ... x Z, so circuits stay in the even sector containing
both the |0...0> initial state and the LMG ground state):

- ``pauli_pair`` with connectivity "all": XX/YY/ZZ PauliRot on every qubit pair
  plus single-qubit Z rotations — the direct all-to-all generalization of the
  SpinGQE reference pool.
- ``pauli_pair`` with connectivity "nn": the reference's nearest-neighbor pool,
  kept as a control.
- ``collective``: exp(-i theta G) for collective generators G in {J_z, J_x^2,
  J_y^2, ...}. These also conserve J^2, so the search stays in the
  (N+1)-dimensional Dicke sector of the exact ground state.

Token convention: token 0 is the BOS/start token; token k >= 1 maps to pool
index k - 1.
"""

import math
from dataclasses import dataclass

import pennylane as qml

from spingqe_lmg.enums import Connectivity, PoolKind

DEFAULT_ANGLES: tuple[float, ...] = tuple(
    s * math.pi / d for s in (1, -1) for d in (2, 4, 8, 16, 32)
)

COLLECTIVE_GENERATORS = ("Jz", "Jx2", "Jy2")


def _collective_generator(name: str, n_qubits: int) -> qml.operation.Operator:
    # Built with queuing suspended: constructing operators inside a QNode would
    # otherwise queue the intermediate Sum/Prod composites onto the tape, where
    # the device applies them as (non-unitary) gates and the state norm explodes.
    with qml.QueuingManager.stop_recording():
        wires = range(n_qubits)

        def j(pauli):
            return qml.s_prod(0.5, qml.sum(*(pauli(w) for w in wires)))

        if name == "Jz":
            return j(qml.PauliZ)
        if name == "Jz2":
            jz = j(qml.PauliZ)
            return qml.prod(jz, jz)
        if name == "Jx2":
            jx = j(qml.PauliX)
            return qml.prod(jx, jx)
        if name == "Jy2":
            jy = j(qml.PauliY)
            return qml.prod(jy, jy)
    raise ValueError(f"unknown collective generator: {name!r}")


def _collective_factors(
    name: str, n_qubits: int
) -> tuple[list[tuple[str, tuple[int, ...]]], float]:
    """Exact factorization data for exp(-i theta G): (Pauli factors, phase/theta).

    J_a^2 = N/4 + (1/2) sum_{i<j} P_i P_j with all terms commuting, so
    exp(-i theta J_a^2) = exp(-i theta N/4) prod PauliRot(theta, PP);
    J_z = (1/2) sum Z_i gives exp(-i theta J_z) = prod PauliRot(theta, Z_i).
    The returned phase is the coefficient of theta in the global phase.
    """
    if name == "Jz":
        return [("Z", (i,)) for i in range(n_qubits)], 0.0
    word = {"Jx2": "XX", "Jy2": "YY", "Jz2": "ZZ"}.get(name)
    if word is None:
        raise ValueError(f"unknown collective generator: {name!r}")
    pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    return [(word, p) for p in pairs], n_qubits / 4


@dataclass(frozen=True)
class PoolOp:
    """One pool entry; ``label`` is the human-readable token name."""

    label: str
    kind: str  # "pauli_rot" | "collective"
    angle: float
    n_qubits: int
    pauli_word: str | None = None
    wires: tuple[int, ...] = ()
    generator: str | None = None

    def factors(self, angle: float | None = None) -> list[tuple[str, tuple[int, ...], float]]:
        """Backend-agnostic gate list: (pauli_word, wires, rotation angle) triples.

        Each triple means exp(-i theta/2 * P) on the given wires (the PauliRot
        convention). Collective ops return their exact commuting-product
        factorization; the global phase is omitted (irrelevant to energies).
        """
        theta = self.angle if angle is None else angle
        if self.kind == "pauli_rot":
            return [(self.pauli_word, tuple(self.wires), theta)]
        if self.kind == "collective":
            word_factors, _ = _collective_factors(self.generator, self.n_qubits)
            return [(word, tuple(ws), theta) for word, ws in word_factors]
        raise ValueError(f"unknown pool op kind: {self.kind!r}")

    def gate(self, angle: float | None = None):
        """Construct (and, inside a QNode, queue) the gate.

        Collective ops are compiled to their *exact* commuting-product form:
        the Pauli terms of J_z and J_a^2 mutually commute, so
        exp(-i theta Jx^2) = exp(-i theta N/4) * prod_{i<j} PauliRot(theta, XX)
        with no Trotter error. This replaces the dense 2^N x 2^N matrix
        exponential (cubic in dimension — the cost cliff that made collective
        N=10 runs take hours) with O(N^2) standard two-qubit kernels.
        Returns a single op for pauli_rot, a list of ops for collective.
        """
        theta = self.angle if angle is None else angle
        if self.kind == "pauli_rot":
            return qml.PauliRot(theta, self.pauli_word, wires=list(self.wires))
        if self.kind == "collective":
            factors, phase = _collective_factors(self.generator, self.n_qubits)
            ops = [qml.PauliRot(theta, word, wires=list(ws)) for word, ws in factors]
            if phase != 0.0:
                ops.append(qml.GlobalPhase(theta * phase))
            return ops
        raise ValueError(f"unknown pool op kind: {self.kind!r}")

    def matrix(self) -> "qml.numpy.ndarray":
        """Dense unitary on the full register (for tests/analysis)."""
        with qml.QueuingManager.stop_recording():
            ops = self.gate()
        if not isinstance(ops, list):
            ops = [ops]
        out = None
        for op in ops:
            mat = qml.matrix(op, wire_order=range(self.n_qubits))
            out = mat if out is None else mat @ out
        return out


def build_pauli_pair_pool(
    n_qubits: int,
    angles: tuple[float, ...] = DEFAULT_ANGLES,
    paulis: tuple[str, ...] = ("ZZ", "XX", "YY"),
    connectivity: Connectivity = Connectivity.ALL,
    include_single_z: bool = True,
) -> list[PoolOp]:
    """Two-qubit PauliRot pool."""
    if isinstance(connectivity, str):
        connectivity = Connectivity(connectivity)
    if connectivity is Connectivity.ALL:
        pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    elif connectivity is Connectivity.NN:
        pairs = [(i, i + 1) for i in range(n_qubits - 1)]
    else:
        raise ValueError(f"unknown connectivity: {connectivity!r}")

    pool = []
    for i, j in pairs:
        for t in angles:
            for word in paulis:
                pool.append(
                    PoolOp(
                        label=f"{word}({i},{j},{t:+.4f})",
                        kind="pauli_rot",
                        angle=t,
                        n_qubits=n_qubits,
                        pauli_word=word,
                        wires=(i, j),
                    )
                )
    if include_single_z:
        for i in range(n_qubits):
            for t in angles:
                pool.append(
                    PoolOp(
                        label=f"Z({i},{t:+.4f})",
                        kind="pauli_rot",
                        angle=t,
                        n_qubits=n_qubits,
                        pauli_word="Z",
                        wires=(i,),
                    )
                )
    return pool


def build_collective_pool(
    n_qubits: int,
    angles: tuple[float, ...] = DEFAULT_ANGLES,
    generators: tuple[str, ...] = COLLECTIVE_GENERATORS,
) -> list[PoolOp]:
    """Pool of exp(-i theta G) for collective-spin generators G."""
    return [
        PoolOp(
            label=f"exp(-i{t:+.4f}*{g})",
            kind="collective",
            angle=t,
            n_qubits=n_qubits,
            generator=g,
        )
        for g in generators
        for t in angles
    ]


def build_pool(cfg, n_qubits: int) -> list[PoolOp]:
    """Dispatch on a PoolConfig (see config.py).

    A nonzero ``cfg.angle_scale`` rescales every pool angle to
    a * angle_scale / n_qubits before dispatch, so the discrete angle set
    tuned at N = angle_scale keeps its effective collective rotation at any N
    (J^2-type generators rotate ~N times faster per unit angle).
    """
    angles = tuple(cfg.angles)
    if cfg.angle_scale:
        angles = tuple(a * cfg.angle_scale / n_qubits for a in angles)
    if cfg.kind is PoolKind.PAULI_PAIR:
        return build_pauli_pair_pool(
            n_qubits,
            angles=angles,
            paulis=tuple(cfg.paulis),
            connectivity=cfg.connectivity,
            include_single_z=cfg.include_single_z,
        )
    if cfg.kind is PoolKind.COLLECTIVE:
        return build_collective_pool(n_qubits, angles=angles, generators=tuple(cfg.generators))
    raise ValueError(f"unknown pool kind: {cfg.kind!r}")
