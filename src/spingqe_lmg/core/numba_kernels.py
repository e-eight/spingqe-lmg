"""Numba-jitted CPU kernels for the statevec/peven sequence-energy evaluators.

Manual index loops and bit-twiddling are intentional at this layer --
vectorized NumPy inside @njit doesn't help, and this is the
performance-critical hot path both IncrementalEvaluator and
ParityEvenEvaluator (evaluators/statevec.py) delegate to.
"""

from enum import IntEnum

import numpy as np
from numba import njit

from spingqe_lmg.enums import InitState


class _InitKind(IntEnum):
    ZERO = 0
    MEAN_FIELD = 1
    GHZ_X = 2


def _resolve_init_kind(init_angle, n_qubits):
    """Return (init_kind: int, theta: float) for the closed-form Numba initial-state builders."""
    if isinstance(init_angle, str):
        if init_angle != InitState.GHZ_X.value:
            raise ValueError(f"unknown reference state: {init_angle!r}")
        if n_qubits % 2:
            raise ValueError("init_state='ghz_x' is defined for even n_qubits")
        return int(_InitKind.GHZ_X), 0.0
    if init_angle == 0.0:
        return int(_InitKind.ZERO), 0.0
    return int(_InitKind.MEAN_FIELD), float(init_angle)


def _flatten_ops(ops_data):
    """Flatten [[(word, wires, theta), ...], ...] into flat arrays for the njit driver.

    Returns (op_offsets, factor_offsets, factor_letters, factor_wires, factor_thetas),
    all as plain lists (converted to typed np arrays by the caller).
    """
    factor_letters, factor_wires, factor_offsets, factor_thetas = [], [], [0], []
    op_offsets = [0]
    for op_factors in ops_data:
        for word_str, wires, theta in op_factors:
            for ltr, w in zip(word_str.encode("ascii"), wires, strict=False):
                factor_letters.append(ltr)
                factor_wires.append(w)
            factor_offsets.append(len(factor_letters))
            factor_thetas.append(theta)
        op_offsets.append(len(factor_thetas))
    return op_offsets, factor_offsets, factor_letters, factor_wires, factor_thetas


def _flatten_offdiag_terms(offdiag_terms, n_qubits=None):
    """Flatten [(coeff, [(wire, letter_byte), ...]), ...] into flat arrays.

    When n_qubits is given, validates wire < n_qubits (raises ValueError otherwise).
    Returns (term_offsets, term_wires, term_letters, coeffs), plain lists.
    """
    term_wires, term_letters, term_offsets, coeffs = [], [], [0], []
    for coeff, letters in offdiag_terms:
        coeffs.append(coeff)
        for w, ltr in letters:
            if n_qubits is not None and w >= n_qubits:
                raise ValueError(f"wire {w} out of range for n_qubits={n_qubits}")
            term_wires.append(w)
            term_letters.append(ltr)
        term_offsets.append(len(term_wires))
    return term_offsets, term_wires, term_letters, coeffs


@njit(cache=True)
def _popcount64(x):
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count


@njit(cache=True)
def _fill_initial_statevec(psi, n_qubits, init_kind, theta):
    if init_kind == 0:  # ZERO
        psi[0] = 1.0
        return
    if init_kind == 1:  # MEAN_FIELD
        c = np.cos(0.5 * theta)
        s = np.sin(0.5 * theta)
        for k in range(psi.shape[0]):
            pc = _popcount64(k)
            psi[k] = c ** (n_qubits - pc) * s**pc
        return
    # GHZ_X
    r = 1.0 / np.sqrt(2.0)
    amp = 2.0 * r ** (n_qubits + 1)
    for k in range(psi.shape[0]):
        psi[k] = amp if (_popcount64(k) % 2 == 0) else 0.0


@njit(cache=True)
def _fill_initial_peven(psi, n_qubits, c_to_full, init_kind, theta):
    if init_kind == 0:  # ZERO
        for k in range(psi.shape[0]):
            psi[k] = 1.0 if c_to_full[k] == 0 else 0.0
        return
    if init_kind == 1:  # MEAN_FIELD
        c = np.cos(0.5 * theta)
        s = np.sin(0.5 * theta)
        for k in range(psi.shape[0]):
            pc = _popcount64(c_to_full[k])
            psi[k] = c ** (n_qubits - pc) * s**pc
        return
    # GHZ_X
    r = 1.0 / np.sqrt(2.0)
    amp = 2.0 * r ** (n_qubits + 1)
    for k in range(psi.shape[0]):
        psi[k] = amp if (_popcount64(c_to_full[k]) % 2 == 0) else 0.0


@njit(cache=True)
def _apply_z_axis(psi, axis):
    stride = 1 << axis
    block = stride << 1
    n = psi.shape[0]
    base = stride
    while base < n:
        for i in range(base, base + stride):
            psi[i] = -psi[i]
        base += block


@njit(cache=True)
def _apply_x_axis(src, dst, axis):
    stride = 1 << axis
    block = stride << 1
    n = src.shape[0]
    base = 0
    while base < n:
        dst[base : base + stride] = src[base + stride : base + block]
        dst[base + stride : base + block] = src[base : base + stride]
        base += block


@njit(cache=True)
def _apply_y_axis(src, dst, axis):
    stride = 1 << axis
    block = stride << 1
    n = src.shape[0]
    base = 0
    while base < n:
        for i in range(stride):
            dst[base + i] = -1j * src[base + stride + i]
        for i in range(stride):
            dst[base + stride + i] = 1j * src[base + i]
        base += block


@njit(cache=True)
def _precompute_terms_sv(term_wires, term_letters, term_offsets, coeffs, n_qubits):
    n_terms = coeffs.shape[0]
    xor_masks = np.zeros(n_terms, dtype=np.int64)
    sign_masks = np.zeros(n_terms, dtype=np.int64)
    c_re = np.zeros(n_terms, dtype=np.float64)
    c_im = np.zeros(n_terms, dtype=np.float64)
    for t in range(n_terms):
        xor_mask = 0
        sign_mask = 0
        num_y = 0
        for i in range(term_offsets[t], term_offsets[t + 1]):
            w = term_wires[i]
            ltr = term_letters[i]
            bit = n_qubits - 1 - w
            if ltr == 88:  # X
                xor_mask ^= 1 << bit
            elif ltr == 89:  # Y
                num_y += 1
                xor_mask ^= 1 << bit
                sign_mask ^= 1 << bit
            elif ltr == 90:  # Z
                sign_mask ^= 1 << bit
        m = num_y % 4
        if m == 0:
            i_re, i_im = 1.0, 0.0
        elif m == 1:
            i_re, i_im = 0.0, 1.0
        elif m == 2:
            i_re, i_im = -1.0, 0.0
        else:
            i_re, i_im = 0.0, -1.0
        xor_masks[t] = xor_mask
        sign_masks[t] = sign_mask
        c_re[t] = coeffs[t] * i_re
        c_im[t] = coeffs[t] * i_im
    return xor_masks, sign_masks, c_re, c_im


@njit(cache=True)
def _fused_energy_sv(psi, h_diag, has_h_diag, identity_coeff, xor_masks, sign_masks, c_re, c_im):
    n = psi.shape[0]
    n_terms = xor_masks.shape[0]
    total = 0.0
    for k in range(n):
        psi_k = psi[k]
        if has_h_diag:
            acc = h_diag[k] * (psi_k.real * psi_k.real + psi_k.imag * psi_k.imag)
        else:
            acc = 0.0
        for t in range(n_terms):
            src = k ^ xor_masks[t]
            psi_src = psi[src]
            if sign_masks[t] != 0 and (_popcount64(k & sign_masks[t]) & 1) == 1:
                sign = -1.0
            else:
                sign = 1.0
            dot_re = psi_k.real * psi_src.real + psi_k.imag * psi_src.imag
            dot_im = psi_k.real * psi_src.imag - psi_k.imag * psi_src.real
            acc += sign * (c_re[t] * dot_re - c_im[t] * dot_im)
        total += acc
    return total + identity_coeff


@njit(cache=True)
def _apply_word_sv(psi, letters, wires, n_qubits, scratch, buf):
    has_non_id = False
    for i in range(letters.shape[0]):
        if letters[i] != 73:  # 'I'
            has_non_id = True
            break
    if not has_non_id:
        scratch[:] = psi
        return 0

    cur_dst_is_scratch = True
    op_idx = 0
    for i in range(letters.shape[0]):
        ltr = letters[i]
        if ltr == 73:
            continue
        axis = n_qubits - 1 - wires[i]
        if op_idx == 0:
            if ltr == 88:
                _apply_x_axis(psi, scratch, axis)
            elif ltr == 89:
                _apply_y_axis(psi, scratch, axis)
            elif ltr == 90:
                scratch[:] = psi
                _apply_z_axis(scratch, axis)
            else:
                scratch[:] = psi
            cur_dst_is_scratch = True
        elif cur_dst_is_scratch:
            if ltr == 88:
                _apply_x_axis(scratch, buf, axis)
            elif ltr == 89:
                _apply_y_axis(scratch, buf, axis)
            elif ltr == 90:
                buf[:] = scratch
                _apply_z_axis(buf, axis)
            else:
                buf[:] = scratch
            cur_dst_is_scratch = False
        else:
            if ltr == 88:
                _apply_x_axis(buf, scratch, axis)
            elif ltr == 89:
                _apply_y_axis(buf, scratch, axis)
            elif ltr == 90:
                scratch[:] = buf
                _apply_z_axis(scratch, axis)
            else:
                scratch[:] = buf
            cur_dst_is_scratch = True
        op_idx += 1

    return 0 if cur_dst_is_scratch else 1


@njit(cache=True)
def _apply_pauli_rot_sv(psi, letters, wires, theta, n_qubits, scratch, buf, next_psi):
    res_idx = _apply_word_sv(psi, letters, wires, n_qubits, scratch, buf)
    ppsi = scratch if res_idx == 0 else buf
    cos_h = np.cos(0.5 * theta)
    sin_h = np.sin(0.5 * theta)
    for i in range(psi.shape[0]):
        next_psi[i] = cos_h * psi[i] - 1j * sin_h * ppsi[i]


@njit(cache=True)
def _statevec_driver(
    n_qubits,
    init_kind,
    theta,
    op_offsets,
    factor_offsets,
    factor_letters,
    factor_wires,
    factor_thetas,
    h_diag,
    has_h_diag,
    identity_coeff,
    xor_masks,
    sign_masks,
    c_re,
    c_im,
):
    n = 1 << n_qubits
    psi = np.zeros(n, dtype=np.complex128)
    _fill_initial_statevec(psi, n_qubits, init_kind, theta)
    scratch = np.zeros(n, dtype=np.complex128)
    buf = np.zeros(n, dtype=np.complex128)
    next_psi = np.zeros(n, dtype=np.complex128)

    n_ops = op_offsets.shape[0] - 1
    energies = np.zeros(n_ops, dtype=np.float64)
    for op_idx in range(n_ops):
        for f in range(op_offsets[op_idx], op_offsets[op_idx + 1]):
            lo = factor_offsets[f]
            hi = factor_offsets[f + 1]
            _apply_pauli_rot_sv(
                psi,
                factor_letters[lo:hi],
                factor_wires[lo:hi],
                factor_thetas[f],
                n_qubits,
                scratch,
                buf,
                next_psi,
            )
            psi, next_psi = next_psi, psi
        energies[op_idx] = _fused_energy_sv(
            psi, h_diag, has_h_diag, identity_coeff, xor_masks, sign_masks, c_re, c_im
        )
    return energies


def statevec_sequence_energies_numba(
    ops_data, n_qubits, init_angle, identity_coeff, h_diag, offdiag_terms
):
    """Numba-jitted statevec sequence-energy kernel.

    ops_data: list of ops in one sequence, each a list of (word_str, wires, theta)
    factors (PoolOp.factors()). Returns one float64 energy per op (prefix energy).
    """
    if len(ops_data) == 0:
        return np.zeros(0, dtype=np.float64)
    init_kind, theta = _resolve_init_kind(init_angle, n_qubits)

    term_offsets, term_wires, term_letters, coeffs = _flatten_offdiag_terms(
        offdiag_terms, n_qubits=n_qubits
    )
    xor_masks, sign_masks, c_re, c_im = _precompute_terms_sv(
        np.array(term_wires, dtype=np.int64),
        np.array(term_letters, dtype=np.int8),
        np.array(term_offsets, dtype=np.int64),
        np.array(coeffs, dtype=np.float64),
        n_qubits,
    )

    op_offsets, factor_offsets, factor_letters, factor_wires, factor_thetas = _flatten_ops(ops_data)

    has_h_diag = h_diag is not None
    h_diag_arr = (
        np.asarray(h_diag, dtype=np.float64) if has_h_diag else np.zeros(1, dtype=np.float64)
    )

    return _statevec_driver(
        n_qubits,
        init_kind,
        theta,
        np.array(op_offsets, dtype=np.int64),
        np.array(factor_offsets, dtype=np.int64),
        np.array(factor_letters, dtype=np.int8),
        np.array(factor_wires, dtype=np.int64),
        np.array(factor_thetas, dtype=np.float64),
        h_diag_arr,
        has_h_diag,
        identity_coeff,
        xor_masks,
        sign_masks,
        c_re,
        c_im,
    )


@njit(cache=True)
def _apply_wire0_z(psi, wire0_bit):
    for k in range(psi.shape[0]):
        if wire0_bit[k]:
            psi[k] = -psi[k]


@njit(cache=True)
def _apply_wire0_y(psi, wire0_bit):
    for k in range(psi.shape[0]):
        if wire0_bit[k]:
            psi[k] = -1j * psi[k]
        else:
            psi[k] = 1j * psi[k]


@njit(cache=True)
def _apply_word_peven(psi, letters, wires, wire0_bit, nc, scratch, buf):
    has_wire0_op = False
    for i in range(letters.shape[0]):
        if wires[i] == 0 and (letters[i] == 90 or letters[i] == 89):
            has_wire0_op = True
            break

    if has_wire0_op:
        buf[:] = psi
        for i in range(letters.shape[0]):
            if wires[i] != 0:
                continue
            if letters[i] == 90:
                _apply_wire0_z(buf, wire0_bit)
            elif letters[i] == 89:
                _apply_wire0_y(buf, wire0_bit)

    has_non_wire0 = False
    for i in range(letters.shape[0]):
        if wires[i] > 0 and letters[i] != 73:
            has_non_wire0 = True
            break

    if not has_non_wire0:
        if has_wire0_op:
            return 1
        scratch[:] = psi
        return 0

    first_src_is_buf = has_wire0_op
    cur_dst_is_scratch = True
    op_idx = 0
    for i in range(letters.shape[0]):
        w = wires[i]
        ltr = letters[i]
        if w == 0 or ltr == 73:
            continue
        axis = nc - w  # msb_first_bit(w-1, nc) == nc - w
        if op_idx == 0:
            src = buf if first_src_is_buf else psi
            if ltr == 88:
                _apply_x_axis(src, scratch, axis)
            elif ltr == 89:
                _apply_y_axis(src, scratch, axis)
            elif ltr == 90:
                scratch[:] = src
                _apply_z_axis(scratch, axis)
            else:
                scratch[:] = src
            cur_dst_is_scratch = True
        elif cur_dst_is_scratch:
            if ltr == 88:
                _apply_x_axis(scratch, buf, axis)
            elif ltr == 89:
                _apply_y_axis(scratch, buf, axis)
            elif ltr == 90:
                buf[:] = scratch
                _apply_z_axis(buf, axis)
            else:
                buf[:] = scratch
            cur_dst_is_scratch = False
        else:
            if ltr == 88:
                _apply_x_axis(buf, scratch, axis)
            elif ltr == 89:
                _apply_y_axis(buf, scratch, axis)
            elif ltr == 90:
                scratch[:] = buf
                _apply_z_axis(scratch, axis)
            else:
                scratch[:] = buf
            cur_dst_is_scratch = True
        op_idx += 1

    return 0 if cur_dst_is_scratch else 1


@njit(cache=True)
def _apply_pauli_rot_peven(psi, letters, wires, theta, wire0_bit, nc, scratch, buf, next_psi):
    res_idx = _apply_word_peven(psi, letters, wires, wire0_bit, nc, scratch, buf)
    ppsi = scratch if res_idx == 0 else buf
    cos_h = np.cos(0.5 * theta)
    sin_h = np.sin(0.5 * theta)
    for i in range(psi.shape[0]):
        next_psi[i] = cos_h * psi[i] - 1j * sin_h * ppsi[i]


@njit(cache=True)
def _precompute_terms_peven(term_wires, term_letters, term_offsets, coeffs, nc):
    n_terms = coeffs.shape[0]
    xor_masks = np.zeros(n_terms, dtype=np.int64)
    sign_masks = np.zeros(n_terms, dtype=np.int64)
    wire0_signs = np.zeros(n_terms, dtype=np.bool_)
    c_re = np.zeros(n_terms, dtype=np.float64)
    c_im = np.zeros(n_terms, dtype=np.float64)
    for t in range(n_terms):
        xor_mask = 0
        sign_mask = 0
        wire0_sign = False
        num_y = 0
        for i in range(term_offsets[t], term_offsets[t + 1]):
            w = term_wires[i]
            ltr = term_letters[i]
            if ltr == 88:  # X
                if w > 0:
                    xor_mask ^= 1 << (nc - w)
            elif ltr == 89:  # Y
                num_y += 1
                if w > 0:
                    xor_mask ^= 1 << (nc - w)
                    sign_mask ^= 1 << (nc - w)
                else:
                    wire0_sign = True
            elif ltr == 90:  # Z
                if w > 0:
                    sign_mask ^= 1 << (nc - w)
                else:
                    wire0_sign = True
        m = num_y % 4
        if m == 0:
            i_re, i_im = 1.0, 0.0
        elif m == 1:
            i_re, i_im = 0.0, 1.0
        elif m == 2:
            i_re, i_im = -1.0, 0.0
        else:
            i_re, i_im = 0.0, -1.0
        xor_masks[t] = xor_mask
        sign_masks[t] = sign_mask
        wire0_signs[t] = wire0_sign
        c_re[t] = coeffs[t] * i_re
        c_im[t] = coeffs[t] * i_im
    return xor_masks, sign_masks, wire0_signs, c_re, c_im


@njit(cache=True)
def _fused_energy_peven(
    psi,
    h_diag,
    has_h_diag,
    identity_coeff,
    xor_masks,
    sign_masks,
    wire0_signs,
    c_re,
    c_im,
    wire0_bit,
):
    n = psi.shape[0]
    n_terms = xor_masks.shape[0]
    total = 0.0
    for k in range(n):
        psi_k = psi[k]
        if has_h_diag:
            acc = h_diag[k] * (psi_k.real * psi_k.real + psi_k.imag * psi_k.imag)
        else:
            acc = 0.0
        if n_terms > 0:
            w0_sign = -1.0 if wire0_bit[k] else 1.0
            for t in range(n_terms):
                src = k ^ xor_masks[t]
                psi_src = psi[src]
                if sign_masks[t] != 0 and (_popcount64(k & sign_masks[t]) & 1) == 1:
                    z_sign = -1.0
                else:
                    z_sign = 1.0
                sign = z_sign * w0_sign if wire0_signs[t] else z_sign
                dot_re = psi_k.real * psi_src.real + psi_k.imag * psi_src.imag
                dot_im = psi_k.real * psi_src.imag - psi_k.imag * psi_src.real
                acc += sign * (c_re[t] * dot_re - c_im[t] * dot_im)
        total += acc
    return total + identity_coeff


@njit(cache=True)
def _peven_driver(
    nc,
    c_to_full,
    init_kind,
    theta,
    n_qubits,
    op_offsets,
    factor_offsets,
    factor_letters,
    factor_wires,
    factor_thetas,
    wire0_bit,
    h_diag,
    has_h_diag,
    identity_coeff,
    xor_masks,
    sign_masks,
    wire0_signs,
    c_re,
    c_im,
):
    half = 1 << nc
    psi = np.zeros(half, dtype=np.complex128)
    _fill_initial_peven(psi, n_qubits, c_to_full, init_kind, theta)
    scratch = np.zeros(half, dtype=np.complex128)
    buf = np.zeros(half, dtype=np.complex128)
    next_psi = np.zeros(half, dtype=np.complex128)

    n_ops = op_offsets.shape[0] - 1
    energies = np.zeros(n_ops, dtype=np.float64)
    for op_idx in range(n_ops):
        for f in range(op_offsets[op_idx], op_offsets[op_idx + 1]):
            lo = factor_offsets[f]
            hi = factor_offsets[f + 1]
            _apply_pauli_rot_peven(
                psi,
                factor_letters[lo:hi],
                factor_wires[lo:hi],
                factor_thetas[f],
                wire0_bit,
                nc,
                scratch,
                buf,
                next_psi,
            )
            psi, next_psi = next_psi, psi
        energies[op_idx] = _fused_energy_peven(
            psi,
            h_diag,
            has_h_diag,
            identity_coeff,
            xor_masks,
            sign_masks,
            wire0_signs,
            c_re,
            c_im,
            wire0_bit,
        )
    return energies


def peven_sequence_energies_numba(
    ops_data, n_qubits, init_angle, c_to_full, wire0_bit, identity_coeff, h_diag, offdiag_terms
):
    """Numba-jitted peven sequence-energy kernel.

    Operates on the compressed (N-1)-qubit even-parity sector. wire 0 is the
    parity wire, handled via wire0_bit phase lookups (never permutes the
    compressed index).
    """
    if len(ops_data) == 0:
        return np.zeros(0, dtype=np.float64)
    init_kind, theta = _resolve_init_kind(init_angle, n_qubits)

    nc = n_qubits - 1

    term_offsets, term_wires, term_letters, coeffs = _flatten_offdiag_terms(
        offdiag_terms, n_qubits=n_qubits
    )
    xor_masks, sign_masks, wire0_signs, c_re, c_im = _precompute_terms_peven(
        np.array(term_wires, dtype=np.int64),
        np.array(term_letters, dtype=np.int8),
        np.array(term_offsets, dtype=np.int64),
        np.array(coeffs, dtype=np.float64),
        nc,
    )

    op_offsets, factor_offsets, factor_letters, factor_wires, factor_thetas = _flatten_ops(ops_data)

    has_h_diag = h_diag is not None
    h_diag_arr = (
        np.asarray(h_diag, dtype=np.float64) if has_h_diag else np.zeros(1, dtype=np.float64)
    )
    wire0_bit_arr = np.asarray(wire0_bit, dtype=np.bool_)
    c_to_full_arr = np.asarray(c_to_full)

    return _peven_driver(
        nc,
        c_to_full_arr,
        init_kind,
        theta,
        n_qubits,
        np.array(op_offsets, dtype=np.int64),
        np.array(factor_offsets, dtype=np.int64),
        np.array(factor_letters, dtype=np.int8),
        np.array(factor_wires, dtype=np.int64),
        np.array(factor_thetas, dtype=np.float64),
        wire0_bit_arr,
        h_diag_arr,
        has_h_diag,
        identity_coeff,
        xor_masks,
        sign_masks,
        wire0_signs,
        c_re,
        c_im,
    )
