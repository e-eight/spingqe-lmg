"""Z2 parity classification and parity-even sector precompute.

Shared by ParityEvenEvaluator (CPU, evaluators/statevec.py) and
ParityEvenGPUEvaluator (evaluators/statevec_gpu.py) -- one implementation,
not two.
"""

from functools import lru_cache

import numpy as np

from spingqe_lmg.core.statevec_ops import initial_statevector
from spingqe_lmg.pools import PoolOp


def pauli_word_parity(word: str) -> int:
    """0 if the Pauli word commutes with Pi = Z x...x Z, 1 if it anticommutes.

    Pi P Pi^-1 = (-1)^(#X + #Y in P) * P, so the parity is the number of
    X/Y letters mod 2.  Words with even parity (ZZ, XX, YY, XY, ...) keep
    a state in its current Z2 sector; odd-parity words (YZ, ZX, single X/Y)
    flip it.
    """
    return sum(1 for c in word if c in "XY") % 2


def pool_parity_summary(pool: list[PoolOp]) -> dict[str, str]:
    """Map each distinct Pauli word in *pool* to ``'even'`` or ``'odd'``.

    Collective ops are expanded to their constituent Pauli words via
    ``PoolOp.factors(angle=0.0)``.  Useful for diagnosing whether a pool
    will keep the circuit in a definite Z2 parity sector.
    """
    words: dict[str, str] = {}
    for op in pool:
        for word, _wires, _theta in op.factors(angle=0.0):
            if word not in words:
                words[word] = "even" if pauli_word_parity(word) == 0 else "odd"
    return words


# ---------------------------------------------------------------------------
# Parity-even subspace utilities
# (moved from evaluators/_peven_hamiltonian.py and evaluators/peven.py)
# ---------------------------------------------------------------------------


_OFFDIAG_PERM_MAX_BYTES = 32 * 1024**2


@lru_cache(maxsize=2)
def build_even_table(n_qubits: int) -> tuple:
    """Precompute compressed<->full index tables for the even-parity sector.

    Wire 0 (the MSB) is the parity qubit: its value is forced to make the total
    qubit-string popcount even.  Compressed index k encodes wires 1..N-1 (the
    lower N-1 bits of the full index); wire 0 is not stored explicitly.
    Full index: ``c_to_full[k] = k | (2^(N-1) if wire0_bit[k] else 0)``.

    Returns
    -------
    c_to_full : int32, shape (2^(N-1),)
        Compressed -> full index table for the even-parity sector.
        Values are full basis indices (up to 2^N-1); int32 is safe for the
        practical memory limit N≤28 (2^28-1 ≈ 268M < 2^31-1).
    full_to_c : int32, shape (2^N,)
        Full -> compressed index table; entries for odd-parity full indices are -1.
    wire0_bit : bool, shape (2^(N-1),)
        ``wire0_bit[k] = popcount(k) % 2``.  True when wire 0 must be |1> to
        make the total popcount even.
    """
    half = 2 ** (n_qubits - 1)
    ks = np.arange(half, dtype=np.int32)
    popcounts = np.bitwise_count(ks)
    wire0_bit = (popcounts % 2).astype(bool)
    msb = np.int32(1) << (n_qubits - 1)
    c_to_full = np.where(popcounts % 2 == 0, ks, ks | msb)
    full_to_c = np.full(2**n_qubits, -1, dtype=np.int32)
    full_to_c[c_to_full] = np.arange(half, dtype=np.int32)
    wire0_bit.flags.writeable = False
    c_to_full.flags.writeable = False
    full_to_c.flags.writeable = False
    return c_to_full, full_to_c, wire0_bit


def _all_distinct_pauli_words(ops) -> list:
    """Return all distinct (word, wires) pairs across ops (PoolOp list)."""
    seen: set = set()
    pairs = []
    for op in ops:
        for word, wires, _theta in op.factors(angle=0.0):
            key = (word, tuple(wires))
            if key not in seen:
                seen.add(key)
                pairs.append((word, tuple(wires)))
    return pairs


def _ham_terms_to_pairs(terms: list) -> list:
    """Extract distinct (word, wires) pairs from Hamiltonian Pauli terms."""
    seen: set = set()
    pairs = []
    for _coeff, letters in terms:
        if not letters:
            continue
        word = "".join(ltr for _w, ltr in letters)
        wires = tuple(w for w, _ltr in letters)
        key = (word, wires)
        if key not in seen:
            seen.add(key)
            pairs.append((word, wires))
    return pairs


def build_h_diag(terms: list, n_qubits: int, wire0_bit: np.ndarray) -> tuple:
    """Split H terms into identity, Z-diagonal, and off-diagonal components."""
    half = 1 << (n_qubits - 1)
    nc = n_qubits - 1
    ks = np.arange(half, dtype=np.int64)

    identity_coeff = 0.0
    h_diag = np.zeros(half, dtype=np.float64)
    offdiag_terms = []

    for coeff, letters in terms:
        if not letters:
            identity_coeff += coeff
            continue
        if any(ltr in ("X", "Y") for _w, ltr in letters):
            offdiag_terms.append((coeff, letters))
            continue
        phase_vec = np.ones(half, dtype=np.float64)
        for w, _ltr in letters:
            if w == 0:
                phase_vec *= np.where(wire0_bit, -1.0, 1.0)
            else:
                bit_val = (ks >> (nc - w)) & 1
                phase_vec *= np.where(bit_val.astype(bool), -1.0, 1.0)
        h_diag += coeff * phase_vec

    h_diag.flags.writeable = False
    return identity_coeff, h_diag, offdiag_terms


def build_offdiag_perms(offdiag_terms: list, n_qubits: int, wire0_bit: np.ndarray):
    """Precompute permutation/phase arrays for off-diagonal Hamiltonian terms.

    Returns None if the required memory exceeds _OFFDIAG_PERM_MAX_BYTES.
    Otherwise returns (perms, phases, coeffs).
    """
    K = len(offdiag_terms)
    if K == 0:
        return np.empty((0, 0), dtype=np.int32), None, np.empty(0, dtype=np.float64)

    half = 1 << (n_qubits - 1)
    nc = n_qubits - 1

    if K * half * 4 > _OFFDIAG_PERM_MAX_BYTES:
        return None

    ks = np.arange(half, dtype=np.int32)
    perms = np.empty((K, half), dtype=np.int32)
    phases_list = []
    coeffs = np.empty(K, dtype=np.float64)
    has_nontrivial_phase = False

    for idx, (coeff, letters) in enumerate(offdiag_terms):
        coeffs[idx] = coeff
        xor_mask = np.int32(0)
        for w, ltr in letters:
            if ltr in ("X", "Y") and w > 0:
                xor_mask ^= np.int32(1 << (nc - w))
        perms[idx] = ks ^ xor_mask

        y_wires = [w for w, ltr in letters if ltr == "Y"]
        num_y = len(y_wires)
        if num_y % 2 != 0:
            raise NotImplementedError(
                f"Hamiltonian term with odd Y-count (num_y={num_y}) is not supported "
                "by the permutation path. The current implementation takes "
                "(1j**num_y).real which is 0 for odd num_y, silently zeroing the term. "
                "The standard LMG Hamiltonian only has num_y in {0, 2}. "
                "To support odd-Y terms, the accumulation loop must use Im(...) "
                "instead of Re(...) for num_y ≡ 1,3 mod 4."
            )
        if num_y == 0:
            phases_list.append(None)
        else:
            i_pow = float((1j**num_y).real)
            sign = np.ones(half, dtype=np.float64)
            for w in y_wires:
                if w == 0:
                    sign *= np.where(wire0_bit, -1.0, 1.0)
                else:
                    bit_val = (ks >> (nc - w)) & 1
                    sign *= np.where(bit_val.astype(bool), -1.0, 1.0)
            phases_list.append(i_pow * sign)
            has_nontrivial_phase = True

    if has_nontrivial_phase:
        phases = np.empty((K, half), dtype=np.float64)
        for idx, ph in enumerate(phases_list):
            phases[idx] = 1.0 if ph is None else ph
        phases.flags.writeable = False
    else:
        phases = None

    perms.flags.writeable = False
    return perms, phases, coeffs


def validate_peven_preconditions(
    pool, terms: list, n_qubits: int, init_angle, c_to_full: np.ndarray, caller_name: str
) -> None:
    """Assert that psi0, pool, and Hamiltonian all respect the parity-even sector.

    Raises ValueError with a diagnostic message on any violation.
    Must be called with the full-space initial state (before compression).
    """
    psi_check = initial_statevector(n_qubits, init_angle, np.complex64)
    odd_mask = np.ones(2**n_qubits, dtype=bool)
    odd_mask[c_to_full] = False
    odd_norm = float(np.linalg.norm(psi_check[odd_mask]))
    if odd_norm > 1e-6:
        raise ValueError(
            f"{caller_name}: init_angle={init_angle!r} produces a state with "
            f"odd-parity components (odd-sector norm={odd_norm:.3g}). "
            "Use init_angle=0.0 or 'ghz_x', or fall back to IncrementalEvaluator."
        )
    for word, wires in _all_distinct_pauli_words(pool):
        if pauli_word_parity(word) != 0:
            raise ValueError(
                f"{caller_name}: Pauli word {word!r} on wires {wires} is parity-odd; "
                "use IncrementalEvaluator instead."
            )
    for word, wires in _ham_terms_to_pairs(terms):
        if pauli_word_parity(word) != 0:
            raise ValueError(
                f"{caller_name}: Hamiltonian word {word!r} on wires {wires} is parity-odd; "
                f"{caller_name} requires a parity-even Hamiltonian."
            )
