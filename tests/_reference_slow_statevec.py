"""Pure-Python reference implementation of the statevec/peven sequence-energy
computation, used ONLY as a correctness oracle for the Numba kernels in
tests/test_numba_kernels.py, tests/test_statevec.py, and tests/test_statevec_gpu.py.

Never imported by any spingqe_lmg production module. This is the former
production Python fallback, demoted here once Numba became the sole
production CPU implementation.
"""

from functools import lru_cache

import numpy as np

from spingqe_lmg.core.statevec_ops import apply_pauli, initial_statevector


def apply_pauli_word(psi, word, wires, n_qubits, buf_a=None, buf_b=None):
    if buf_a is not None and buf_b is not None:
        first = buf_b if psi is buf_a else buf_a
        bufs = (first, buf_b if first is buf_a else buf_a)
    src = psi
    k = 0
    for letter, wire in zip(word, wires, strict=True):
        if letter == "I":
            continue
        if buf_a is not None and buf_b is not None:
            dst = bufs[k % 2]
            apply_pauli(src, letter, wire, n_qubits, out=dst)
            src = dst
        else:
            src = apply_pauli(src, letter, wire, n_qubits)
        k += 1
    return src


def apply_pauli_rot(psi, word, wires, theta, n_qubits, buf_a=None, buf_b=None, out=None):
    if out is not None:
        if buf_a is not None and out is buf_a:
            raise ValueError(
                "apply_pauli_rot: out must not alias buf_a (aliased scratch buffer "
                "causes in-place clobber before the add step)"
            )
        if buf_b is not None and out is buf_b:
            raise ValueError(
                "apply_pauli_rot: out must not alias buf_b (aliased scratch buffer "
                "causes in-place clobber before the add step)"
            )
    half = 0.5 * theta
    ppsi = apply_pauli_word(psi, word, wires, n_qubits, buf_a=buf_a, buf_b=buf_b)
    cos_half = np.cos(half)
    sin_half = np.sin(half)
    if out is None:
        out = cos_half * psi - 1j * sin_half * ppsi
    else:
        np.multiply(psi, cos_half, out=out)
        np.multiply(ppsi, -1j * sin_half, out=ppsi)
        np.add(out, ppsi, out=out)
    return out


def reference_statevec_sequence_energies(
    ops_data, terms, n_qubits, init_angle, identity_coeff=0.0, h_diag=None, offdiag_terms=None
):
    """ops_data: list of ops, each a list of (word, wires, theta) factors."""
    from spingqe_lmg.core.statevec_ops import energy_from_terms

    if not ops_data:
        return np.zeros(0)
    psi = initial_statevector(n_qubits, init_angle, np.complex128)
    n2 = 2**n_qubits
    buf_a = np.empty(n2, dtype=psi.dtype)
    buf_b = np.empty(n2, dtype=psi.dtype)
    next_psi = np.empty(n2, dtype=psi.dtype)
    out = []
    for factors in ops_data:
        for word, wires, theta in factors:
            apply_pauli_rot(
                psi, word, wires, theta, n_qubits, buf_a=buf_a, buf_b=buf_b, out=next_psi
            )
            psi, next_psi = next_psi, psi
        out.append(energy_from_terms(psi, terms, n_qubits, buf_a=buf_a, buf_b=buf_b))
    return np.array(out)


@lru_cache(maxsize=32)
def _axis_factor_c(values, axis, nc, dtype_name):
    shape = [1] * nc
    shape[axis] = 2
    arr = np.asarray(values, dtype=np.dtype(dtype_name)).reshape(shape)
    arr.flags.writeable = False
    return arr


def _apply_word_peven(psi, word, wires, n_qubits, wire0_bit, out, buf, dtype_name, wire0_buf):
    nc = n_qubits - 1
    wire0_factor = None
    for letter, wire in zip(word, wires, strict=True):
        if letter == "I" or wire != 0:
            continue
        if wire0_factor is None:
            wire0_buf[:] = 1.0
            wire0_factor = wire0_buf
        if letter == "Z":
            wire0_factor *= np.where(wire0_bit, -1.0, 1.0)
        elif letter == "Y":
            wire0_factor *= np.where(wire0_bit, -1j, 1j)

    if wire0_factor is not None:
        np.multiply(psi, wire0_factor, out=buf)
        src = buf
    else:
        src = psi

    src_t = src.reshape((2,) * nc)
    dst_is_out = True
    k = 0
    for letter, wire in zip(word, wires, strict=True):
        if letter == "I" or wire == 0:
            continue
        axis = wire - 1
        dst = out if dst_is_out else buf
        dst_t = dst.reshape((2,) * nc)
        if letter == "X":
            idx0 = [slice(None)] * nc
            idx1 = [slice(None)] * nc
            idx0[axis] = 0
            idx1[axis] = 1
            dst_t[tuple(idx0)] = src_t[tuple(idx1)]
            dst_t[tuple(idx1)] = src_t[tuple(idx0)]
        elif letter == "Y":
            idx0 = [slice(None)] * nc
            idx1 = [slice(None)] * nc
            idx0[axis] = 0
            idx1[axis] = 1
            dst_t[tuple(idx0)] = src_t[tuple(idx1)]
            dst_t[tuple(idx1)] = src_t[tuple(idx0)]
            dst_t *= _axis_factor_c((-1j, 1j), axis, nc, dtype_name)
        elif letter == "Z":
            np.multiply(src_t, _axis_factor_c((1.0, -1.0), axis, nc, dtype_name), out=dst_t)
        src_t = dst_t
        dst_is_out = not dst_is_out
        k += 1

    if k == 0:
        if src is not out:
            np.copyto(out, src)
        return out
    if dst_is_out:
        np.copyto(out, buf)
    return out


def _apply_pauli_rot_peven(
    psi, word, wires, theta, n_qubits, wire0_bit, scratch, out, dtype_name, wire0_buf
):
    ppsi = _apply_word_peven(
        psi,
        word,
        wires,
        n_qubits,
        wire0_bit,
        out=scratch,
        buf=out,
        dtype_name=dtype_name,
        wire0_buf=wire0_buf,
    )
    cos_h = np.cos(0.5 * theta)
    sin_h = np.sin(0.5 * theta)
    np.multiply(psi, cos_h, out=out)
    np.multiply(ppsi, -1j * sin_h, out=ppsi)
    np.add(out, ppsi, out=out)
    return out


def _energy_from_terms_peven(psi, terms, n_qubits, wire0_bit, scratch, buf, dtype_name, wire0_buf):
    total = 0.0
    for coeff, letters in terms:
        if not letters:
            total += coeff
            continue
        word = "".join(ltr for _w, ltr in letters)
        wires = tuple(w for w, _ltr in letters)
        phi = _apply_word_peven(
            psi,
            word,
            wires,
            n_qubits,
            wire0_bit,
            out=scratch,
            buf=buf,
            dtype_name=dtype_name,
            wire0_buf=wire0_buf,
        )
        total += coeff * np.vdot(psi, phi).real
    return float(total)


def _energy_fast_peven(
    psi,
    identity_coeff,
    h_diag,
    offdiag_terms,
    n_qubits,
    wire0_bit,
    scratch,
    buf,
    dtype_name,
    wire0_buf,
    offdiag_perms,
    offdiag_phases,
    offdiag_coeffs,
    phi=None,
):
    prob = psi.real * psi.real + psi.imag * psi.imag
    e = float(identity_coeff) + np.dot(h_diag, prob)
    if offdiag_perms is not None:
        psi_conj = np.conj(psi)
        K = len(offdiag_perms)
        ips = np.empty(K, dtype=np.float64)
        if phi is None:
            phi = np.empty(len(psi), dtype=psi.dtype)
        if offdiag_phases is not None:
            for k in range(K):
                np.take(psi, offdiag_perms[k], out=phi)
                phi *= offdiag_phases[k]
                ips[k] = np.dot(psi_conj, phi).real
        else:
            for k in range(K):
                np.take(psi, offdiag_perms[k], out=phi)
                ips[k] = np.dot(psi_conj, phi).real
        e += float(offdiag_coeffs @ ips)
    elif offdiag_terms:
        e += _energy_from_terms_peven(
            psi, offdiag_terms, n_qubits, wire0_bit, scratch, buf, dtype_name, wire0_buf
        )
    return float(e)


def reference_peven_sequence_energies(
    ops_data,
    terms,
    n_qubits,
    init_angle,
    c_to_full,
    wire0_bit,
    identity_coeff=0.0,
    h_diag=None,
    offdiag_terms=None,
):
    if not ops_data:
        return np.zeros(0)
    psi_full = initial_statevector(n_qubits, init_angle, np.complex128)
    psi = psi_full[c_to_full].astype(np.complex128)
    half = len(psi)
    scratch = np.empty(half, dtype=psi.dtype)
    next_psi = np.empty(half, dtype=psi.dtype)
    wire0_buf = np.empty(half, dtype=psi.dtype)

    out_list = []
    for factors in ops_data:
        for word, wires, theta in factors:
            _apply_pauli_rot_peven(
                psi,
                word,
                wires,
                theta,
                n_qubits,
                wire0_bit,
                scratch,
                next_psi,
                "complex128",
                wire0_buf,
            )
            psi, next_psi = next_psi, psi
        if h_diag is not None:
            e = _energy_fast_peven(
                psi,
                identity_coeff,
                h_diag,
                offdiag_terms,
                n_qubits,
                wire0_bit,
                scratch,
                next_psi,
                "complex128",
                wire0_buf,
                None,
                None,
                None,
            )
        else:
            e = _energy_from_terms_peven(
                psi, terms, n_qubits, wire0_bit, scratch, next_psi, "complex128", wire0_buf
            )
        out_list.append(e)
    return np.array(out_list)
