"""Dicke-basis simulation of the collective-pool GQE training loop.

The collective pool's gates exp(-i theta G), G in {Jz, Jx^2, Jy^2, Jz2}, the
LMG Hamiltonian, and both reference states (|0...0> = |j, j> and the
mean-field coherent state exp(-i theta_mf Jy)|j, j>) all live in the
permutation-symmetric j = N/2 sector. Simulating there reduces every
operation from 2^N to (N+1) dimensions, making the training loop polynomial
in N — the basis of the N = 100-1000 showcase.

This is a classical-simulability statement about permutation-invariant
circuits (Anschuetz et al., Quantum 7, 1189 (2023); Somma et al., PRL 97,
190501 (2006)): the same symmetry that makes the collective pool trainable
makes it classically verifiable at scale.

Analytic expectation values only (no shots path); refinement runs L-BFGS on
the (N+1)-dim cost directly.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from spingqe_lmg.core.caching import CachedEvaluatorMixin, LRUCache
from spingqe_lmg.exact import collective_spin_matrices, dicke_matrix
from spingqe_lmg.pools import PoolOp
from spingqe_lmg.refine import RefineResult


@dataclass(frozen=True)
class DickeSpace:
    """Eigendecomposed collective generators and the Hamiltonian, j = N/2 sector.

    Operates in the permutation-symmetric (Dicke) sector; compatible with the
    collective pool only.

    The Hamiltonian stored here includes the constant term lambda(1+gamma)/4 that arises
    from expanding sum_{i!=j} sigma_a sigma_a = 4J_a^2 - N.  This term is absent from the
    qubit-level Hamiltonian in ``hamiltonians.lmg_hamiltonian`` but is algebraically
    equivalent.  See ``exact.dicke_matrix`` for the derivation.
    All energies returned by this class are total <H> (not per-spin).
    """

    n_qubits: int
    ham: np.ndarray
    gen_eigvals: dict[str, np.ndarray]
    gen_eigvecs: dict[str, np.ndarray]
    ham_vals: np.ndarray
    ham_vecs: np.ndarray

    @classmethod
    def build(cls, n_qubits: int, h: float, lam: float, gamma: float = 0.0) -> "DickeSpace":
        jx, jy, jz = collective_spin_matrices(n_qubits)
        generators = {"Jz": jz, "Jx2": jx @ jx, "Jy2": jy @ jy, "Jz2": jz @ jz}
        vals, vecs = {}, {}
        for name, gen in generators.items():
            w, v = np.linalg.eigh(gen)
            vals[name], vecs[name] = w, v
        ham = dicke_matrix(n_qubits, h, lam, gamma).astype(complex)
        ham.flags.writeable = False
        for arr in vals.values():
            arr.flags.writeable = False
        for arr in vecs.values():
            arr.flags.writeable = False
        ham_vals, ham_vecs = np.linalg.eigh(ham)
        ham_vals.flags.writeable = False
        ham_vecs.flags.writeable = False
        return cls(
            n_qubits=n_qubits,
            ham=ham,
            gen_eigvals=vals,
            gen_eigvecs=vecs,
            ham_vals=ham_vals,
            ham_vecs=ham_vecs,
        )

    def initial_state(self, init_angle: float = 0.0) -> np.ndarray:
        """|j, j> (index 0: m ordered j..-j), or the coherent state RY(theta)^N."""
        if isinstance(init_angle, str):
            # ghz_x lives in this sector ((|j,mx=j> + |j,mx=-j>)/sqrt2) but is
            # not implemented here yet; the GHZ pilot uses the qubit path.
            raise NotImplementedError(
                "the dicke evaluator supports only product-state (RY angle) references"
            )
        psi = np.zeros(self.n_qubits + 1, dtype=complex)
        psi[0] = 1.0
        if init_angle != 0.0:
            # product of single-qubit RY(theta) == exp(-i theta Jy) on the sector
            _, jy, _ = collective_spin_matrices(self.n_qubits)
            w, v = np.linalg.eigh(jy)
            psi = v @ (np.exp(-1j * init_angle * w) * (v.conj().T @ psi))
        return psi

    def apply_gate(self, psi: np.ndarray, generator: str, theta: float) -> np.ndarray:
        w, v = self.gen_eigvals[generator], self.gen_eigvecs[generator]
        return v @ (np.exp(-1j * theta * w) * (v.conj().T @ psi))

    def energy(self, psi: np.ndarray) -> float:
        return float(np.real(psi.conj() @ self.ham @ psi))


def _check_collective(pool: list[PoolOp]) -> None:
    bad = [op.label for op in pool if op.kind != "collective"]
    if bad:
        raise ValueError(f"the Dicke evaluator supports collective pools only; got {bad[:3]}...")


class DickeEvaluator(CachedEvaluatorMixin):
    """Dicke-sector (permutation-symmetric) evaluator for the collective pool.

    Operates in the (N+1)-dimensional Dicke basis, avoiding the exponential
    cost of full statevector simulation.  Called the "Dicke sector evaluator"
    or "symmetric sector evaluator" in the manuscripts.  Compatible with the
    collective pool only; raises ValueError for pairwise pools.
    """

    def __init__(
        self,
        pool: list[PoolOp],
        n_qubits: int,
        h: float,
        lam: float,
        gamma: float = 0.0,
        init_angle: float = 0.0,
        n_jobs: int = 1,  # interface parity; (N+1)-dim work doesn't need it
        cache_maxsize: int | None = None,
    ):
        _check_collective(pool)
        self.pool = list(pool)
        self.space = DickeSpace.build(n_qubits, h, lam, gamma)
        self.init_angle = init_angle
        self._psi0 = self.space.initial_state(init_angle)
        self._cache: LRUCache = LRUCache(cache_maxsize)

    def initial_energy(self) -> float:
        return self.space.energy(self._psi0)

    @property
    def ground_energy(self) -> float:
        """Exact ground-state energy from the DickeSpace eigendecomposition.

        Already computed by DickeSpace.build(); no additional diagonalisation needed.
        """
        return float(self.space.ham_vals[0])

    def _sequence_energies(self, key: tuple[int, ...]) -> np.ndarray:
        psi = self._psi0
        energies = []
        for i in key:
            op = self.pool[i]
            psi = self.space.apply_gate(psi, op.generator, op.angle)
            energies.append(self.space.energy(psi))
        return np.array(energies)

    def refine_tokens(
        self,
        tokens: list[int],
        *,
        qml_device: str = "",
        shots: int = 0,
        seed: int = 0,
        optimizer=None,
        gradient=None,
    ) -> RefineResult:
        return dicke_refine_tokens(tokens, self.pool, self.space, init_angle=self.init_angle)


def dicke_refine_tokens(
    tokens: list[int],
    pool: list[PoolOp],
    space: DickeSpace,
    init_angle: float = 0.0,
    x0: Sequence[float] | None = None,
) -> RefineResult:
    """L-BFGS angle refinement of a collective sequence in the Dicke sector.

    ``x0`` overrides the starting angles (default: each op's pool angle), so
    callers can start from random angles instead of the discrete token grid.
    """
    ops = [pool[t - 1] for t in tokens[1:]]
    _check_collective(ops)
    psi0 = space.initial_state(init_angle)

    def cost(params: np.ndarray) -> float:
        psi = psi0
        for op, theta in zip(ops, params, strict=True):
            psi = space.apply_gate(psi, op.generator, float(theta))
        return space.energy(psi)

    x0 = np.array([op.angle for op in ops] if x0 is None else x0, dtype=float)
    initial = cost(x0)
    res = minimize(cost, x0, method="L-BFGS-B", tol=1e-12)
    return RefineResult(
        energy=float(res.fun),
        angles=[float(a) for a in res.x],
        initial_energy=float(initial),
        n_evaluations=int(res.nfev),
    )


def dicke_state_metrics(
    tokens: list[int],
    angles: list[float] | None,
    pool: list[PoolOp],
    space: DickeSpace,
    init_angle: float = 0.0,
) -> dict:
    """Fidelity with the exact sector ground state + order parameter, any N."""
    if any(t == 0 for t in tokens[1:]):
        raise ValueError(f"BOS token at non-zero position in tokens: {tokens}")
    ops = [pool[t - 1] for t in tokens[1:]]
    psi = space.initial_state(init_angle)
    thetas = angles if angles is not None else [op.angle for op in ops]
    for op, theta in zip(ops, thetas, strict=True):
        psi = space.apply_gate(psi, op.generator, float(theta))
    fidelity = float(np.abs(space.ham_vecs[:, 0].conj() @ psi) ** 2)
    n = space.n_qubits
    jx, _, _ = collective_spin_matrices(n)
    order = float(np.real(psi.conj() @ (jx @ jx) @ psi) * 4 / n**2)
    return {"fidelity_ground": fidelity, "order_parameter": order, "energy": space.energy(psi)}
