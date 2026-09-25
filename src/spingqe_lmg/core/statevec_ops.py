"""Matrix-free Pauli operations and Hamiltonian decomposition on 2^N statevectors.

Shared by the CPU and GPU incremental/parity-even evaluators.
"""

from functools import cache, reduce

import numpy as np


@cache
def _axis_factor(values: tuple, wire: int, n_qubits: int, dtype_name: str) -> np.ndarray:
    """Length-2 factor broadcast along one qubit axis of the reshaped state.

    Results are cached (keyed by values/wire/n_qubits/dtype) and marked
    read-only so they are safe to return from cache without copying.
    """
    shape = [1] * n_qubits
    shape[wire] = 2
    arr = np.asarray(values, dtype=np.dtype(dtype_name)).reshape(shape)
    arr.flags.writeable = False
    return arr


def apply_pauli(
    psi: np.ndarray,
    pauli: str,
    wire: int,
    n_qubits: int,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Return (P_wire |psi>) for P in {X, Y, Z}; O(2^N) time.

    When *out* is provided the result is written into it in-place (zero-copy
    return). The input *psi* is never modified.
    """
    psi = np.asarray(psi)
    t = psi.reshape((2,) * n_qubits)
    if pauli == "X":
        if out is None:
            return np.flip(t, axis=wire).reshape(psi.shape)
        out_t = out.reshape((2,) * n_qubits)
        idx0: list = [slice(None)] * n_qubits
        idx1: list = [slice(None)] * n_qubits
        idx0[wire] = 0
        idx1[wire] = 1
        out_t[tuple(idx0)] = t[tuple(idx1)]
        out_t[tuple(idx1)] = t[tuple(idx0)]
        return out
    elif pauli == "Y":
        dtype = np.result_type(psi.dtype, np.complex64)
        if out is None:
            flipped = np.flip(t, axis=wire)
            return (flipped * _axis_factor((-1j, 1j), wire, n_qubits, dtype.name)).reshape(
                psi.shape
            )
        out_t = out.reshape((2,) * n_qubits)
        idx0: list = [slice(None)] * n_qubits
        idx1: list = [slice(None)] * n_qubits
        idx0[wire] = 0
        idx1[wire] = 1
        out_t[tuple(idx0)] = t[tuple(idx1)]
        out_t[tuple(idx1)] = t[tuple(idx0)]
        np.multiply(out_t, _axis_factor((-1j, 1j), wire, n_qubits, dtype.name), out=out_t)
        return out
    elif pauli == "Z":
        if out is None:
            return (t * _axis_factor((1, -1), wire, n_qubits, psi.dtype.name)).reshape(psi.shape)
        out_t = out.reshape((2,) * n_qubits)
        np.multiply(t, _axis_factor((1, -1), wire, n_qubits, psi.dtype.name), out=out_t)
        return out
    else:
        raise ValueError(f"unknown Pauli: {pauli!r}")


def apply_jx(psi: np.ndarray, n_qubits: int) -> np.ndarray:
    """(J_x |psi>) with J_x = (1/2) sum_i X_i; O(N 2^N) time, O(2^N) memory."""
    out = apply_pauli(psi, "X", 0, n_qubits)
    for w in range(1, n_qubits):
        out += apply_pauli(psi, "X", w, n_qubits)
    return 0.5 * out


def apply_jy(psi: np.ndarray, n_qubits: int) -> np.ndarray:
    """(J_y |psi>) with J_y = (1/2) sum_i Y_i."""
    out = apply_pauli(psi, "Y", 0, n_qubits)
    for w in range(1, n_qubits):
        out += apply_pauli(psi, "Y", w, n_qubits)
    return 0.5 * out


def order_param_sq_jx(psi: np.ndarray, n_qubits: int) -> float:
    """<(2 J_x / N)^2> for a normalized state.

    J_x is Hermitian, so <Jx^2> = ||Jx |psi>||^2 --- one matrix-free
    application instead of the dense 2^N x 2^N Jx @ Jx product (the last
    member of the accidental-dense-4^N OOM class).
    """
    jpsi = apply_jx(psi, n_qubits)
    return float(np.real(np.vdot(jpsi, jpsi))) * 4 / n_qubits**2


# --- incremental prefix evaluator --------------------------------------


def hamiltonian_terms(ham) -> list[tuple[float, tuple[tuple[int, str], ...]]]:
    """Flatten a Pauli-sum qml.Hamiltonian to (coeff, ((wire, letter), ...)) terms.

    Uses op.pauli_rep so sum-valued ops (e.g. the Heisenberg XX+YY+ZZ bond) are
    expanded into their individual Pauli words. The empty word () is identity.
    """
    coeffs, ops = ham.terms()
    terms: list[tuple[float, tuple[tuple[int, str], ...]]] = []
    for coeff, op in zip(coeffs, ops, strict=True):
        sentence = op.pauli_rep
        if sentence is None:
            raise ValueError("IncrementalEvaluator requires a Pauli-sum Hamiltonian")
        for pauli_word, inner in sentence.items():
            letters = tuple(sorted(pauli_word.items()))  # ((wire, 'X'), ...)
            terms.append((float(np.real(complex(coeff) * complex(inner))), letters))
    return terms


def _build_sv_h_diag(terms, n_qubits: int) -> np.ndarray:
    """Diagonal Hamiltonian array for Z-only terms, shape (2^N,).

    Wire w in statevec.py uses MSB-first reshape: wire 0 = axis 0 = bit N-1 of
    flat index.  Z eigenvalue at wire w for basis state k:
    +1 if bit (N-1-w) of k is 0, -1 if bit (N-1-w) of k is 1.
    """
    n = 2**n_qubits
    h = np.zeros(n, dtype=np.float64)
    indices = np.arange(n, dtype=np.int64)
    for coeff, letters in terms:
        if not letters:
            continue
        if all(ltr == "Z" for _, ltr in letters):
            factor = np.ones(n, dtype=np.float64)
            for wire, _ in letters:
                bit_pos = n_qubits - 1 - wire  # wire 0 = MSB = bit N-1
                factor *= np.where(indices & (1 << bit_pos), -1.0, 1.0)
            h += coeff * factor
    return h


def split_sv_terms(terms, n_qubits: int):
    """Split Hamiltonian terms into (identity_coeff, h_diag, offdiag_terms).

    offdiag_terms: [(coeff, [(wire, letter_byte)])] where letter_byte is ASCII
    (X=88, Y=89, Z=90) --- the format expected by statevec_sequence_energies_numba.
    """
    identity_coeff = sum(float(c) for c, ls in terms if not ls)
    offdiag = [
        (float(coeff), [(int(w), ord(ltr)) for w, ltr in letters])
        for coeff, letters in terms
        if letters and any(ltr in "XY" for _, ltr in letters)
    ]
    h_diag = _build_sv_h_diag(terms, n_qubits)
    return identity_coeff, h_diag, offdiag


def energy_from_terms(
    psi: np.ndarray,
    terms: list[tuple[float, tuple[tuple[int, str], ...]]],
    n_qubits: int,
    buf_a: np.ndarray | None = None,
    buf_b: np.ndarray | None = None,
) -> float:
    """<psi|H|psi> from pre-decomposed Pauli terms; matrix-free, O(terms * 2^N).

    When *buf_a* and *buf_b* are provided, intermediate Pauli-application
    results are written into them instead of being freshly allocated, keeping
    GC pressure constant across arbitrarily many Hamiltonian terms.
    """
    total = 0.0
    for coeff, letters in terms:
        if not letters:  # identity term
            total += coeff * np.vdot(psi, psi).real
        else:
            phi = psi
            k = 0
            for wire, letter in letters:
                if buf_a is not None and buf_b is not None:
                    dst = buf_a if k % 2 == 0 else buf_b
                    apply_pauli(phi, letter, wire, n_qubits, out=dst)
                    phi = dst
                else:
                    phi = apply_pauli(phi, letter, wire, n_qubits)
                k += 1
            total += coeff * np.vdot(psi, phi).real
    return float(total)


def initial_statevector(
    n_qubits: int, init_angle: float | str = 0.0, dtype=np.complex128
) -> np.ndarray:
    """Reference statevector matching energy.prepare_initial_state (qml ordering).

    zero -> |0...0>; float theta -> product RY(theta)^N; "ghz_x" ->
    (|+...+> + |-...->)/sqrt(2), the parity-even GHZ-x stabilizer state.
    """
    if isinstance(init_angle, str):
        if init_angle != "ghz_x":
            raise ValueError(f"unknown reference state: {init_angle!r}")
        if n_qubits % 2:
            raise ValueError("init_state='ghz_x' is defined for even n_qubits")
        r = 1.0 / np.sqrt(2.0)
        plus = reduce(np.kron, [np.array([r, r], dtype=complex)] * n_qubits)
        minus = reduce(np.kron, [np.array([r, -r], dtype=complex)] * n_qubits)
        psi = (plus + minus) * r
    elif init_angle == 0.0:
        psi = np.zeros(2**n_qubits, dtype=complex)
        psi[0] = 1.0
    else:
        c, s = np.cos(init_angle / 2), np.sin(init_angle / 2)
        psi = reduce(np.kron, [np.array([c, s], dtype=complex)] * n_qubits)
    return psi.astype(dtype)
