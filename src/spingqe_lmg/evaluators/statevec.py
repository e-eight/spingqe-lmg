"""Incremental full-Hilbert statevector evaluator (CPU, Python fallback).

Steps a held 2^N statevector gate-by-gate, computing per-prefix energies in
O(L) applications per sequence.  Optional ``complex64`` dtype for
throughput-sensitive scaling runs.
"""

import warnings

import numpy as np

from spingqe_lmg.core.caching import CachedEvaluatorMixin, LRUCache
from spingqe_lmg.core.numba_kernels import (
    peven_sequence_energies_numba,
    statevec_sequence_energies_numba,
)
from spingqe_lmg.core.parity import (
    build_even_table,
    build_h_diag,
    build_offdiag_perms,
    validate_peven_preconditions,
)
from spingqe_lmg.core.statevec_ops import (
    split_sv_terms,
    energy_from_terms,
    hamiltonian_terms,
    initial_statevector,
)
from spingqe_lmg.pools import PoolOp
from spingqe_lmg.refine import RefineResult, refine_tokens

# --- main evaluator ----------------------------------------------------


class IncrementalEvaluator(CachedEvaluatorMixin):
    """Incremental statevector evaluator (CPU, full 2^N Hilbert space).

    Steps a held statevector gate-by-gate — one gate application per pool op,
    O(L) applications per sequence.  Called the "incremental evaluator" in the
    manuscript.  For the parity-even subspace variant see ParityEvenEvaluator.
    """

    def __init__(
        self,
        pool: list[PoolOp],
        ham,
        n_qubits: int,
        init_angle: float | str = 0.0,
        n_jobs: int = 1,
        dtype=np.complex128,
        cache_maxsize: int | None = None,
    ):
        dtype = np.dtype(dtype)
        if dtype == np.complex64:
            warnings.warn(
                "dtype=complex64 is no longer supported on CPU (Numba computes in "
                "complex128); results are unaffected but memory/throughput won't "
                "reflect complex64.",
                RuntimeWarning,
                stacklevel=2,
            )
            dtype = np.dtype(np.complex128)
        self.pool = list(pool)
        self.ham = ham
        self.n_qubits = n_qubits
        self.init_angle = init_angle
        self.n_jobs = n_jobs
        self.dtype = dtype
        self.terms = hamiltonian_terms(ham)
        self._cache: LRUCache = LRUCache(cache_maxsize)
        self._identity_coeff, self.h_diag, self.offdiag_terms = split_sv_terms(
            self.terms, n_qubits
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_cache", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache = LRUCache(None)

    def initial_energy(self) -> float:
        psi = initial_statevector(self.n_qubits, self.init_angle, self.dtype)
        return energy_from_terms(psi, self.terms, self.n_qubits)

    def _ops_for(self, key: tuple[int, ...]) -> tuple[PoolOp, ...]:
        return tuple(self.pool[i] for i in key)

    def _sequence_energies(self, key: tuple[int, ...]) -> np.ndarray:
        ops = self._ops_for(key)
        if not ops:
            return np.zeros(0)
        ops_data = [
            [(word, list(wires), float(theta)) for word, wires, theta in op.factors()] for op in ops
        ]
        return statevec_sequence_energies_numba(
            ops_data,
            self.n_qubits,
            self.init_angle,
            self._identity_coeff,
            self.h_diag,
            self.offdiag_terms,
        )

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


# --- parity-even evaluator ----------------------------------------------


class ParityEvenEvaluator(IncrementalEvaluator):
    """Parity-even statevector evaluator (CPU, 2^(N-1) compressed basis).

    Restricts the incremental evaluator to the parity-even (Z_2) sector,
    halving both memory and arithmetic cost.  Called the "parity-even evaluator"
    in the manuscript.  Compatible with parity-even pools only.
    """

    def __init__(
        self,
        pool: list,
        ham,
        n_qubits: int,
        init_angle: float | str = 0.0,
        n_jobs: int = 1,
        dtype=np.complex128,
        cache_maxsize: int | None = None,
    ):
        if n_qubits < 2:
            raise ValueError("ParityEvenEvaluator requires n_qubits >= 2")
        super().__init__(pool, ham, n_qubits, init_angle, n_jobs, dtype, cache_maxsize)

        c_to_full, full_to_c, wire0_bit = build_even_table(n_qubits)
        self._c_to_full = c_to_full

        self._wire0_bit = wire0_bit

        validate_peven_preconditions(
            pool,
            self.terms,
            n_qubits,
            init_angle,
            c_to_full,
            "ParityEvenEvaluator",
        )

        identity_coeff, h_diag, offdiag_terms = build_h_diag(self.terms, n_qubits, wire0_bit)
        self._identity_coeff = identity_coeff
        self._h_diag = h_diag
        self._offdiag_terms = offdiag_terms

        result = build_offdiag_perms(offdiag_terms, n_qubits, wire0_bit)
        if result is None:
            self._offdiag_perms = self._offdiag_phases = self._offdiag_coeffs = None
        else:
            self._offdiag_perms, self._offdiag_phases, self._offdiag_coeffs = result

    def _sequence_energies(self, key: tuple[int, ...]) -> np.ndarray:
        ops = self._ops_for(key)
        if not ops:
            return np.zeros(0)
        ops_data = [
            [(word, list(wires), float(theta)) for word, wires, theta in op.factors()] for op in ops
        ]
        offdiag_terms_encoded = [
            (float(c), [(int(w), ord(ltr)) for w, ltr in ltrs]) for c, ltrs in self._offdiag_terms
        ]
        return peven_sequence_energies_numba(
            ops_data,
            self.n_qubits,
            self.init_angle,
            self._c_to_full,
            self._wire0_bit,
            self._identity_coeff,
            self._h_diag,
            offdiag_terms_encoded,
        )

    def initial_energy(self) -> float:
        """Energy of the bare initial state, via a single no-op identity gate."""
        offdiag_terms_encoded = [
            (float(c), [(int(w), ord(ltr)) for w, ltr in ltrs]) for c, ltrs in self._offdiag_terms
        ]
        energies = peven_sequence_energies_numba(
            [[("I", [1], 0.0)]],
            self.n_qubits,
            self.init_angle,
            self._c_to_full,
            self._wire0_bit,
            self._identity_coeff,
            self._h_diag,
            offdiag_terms_encoded,
        )
        return float(energies[0])
