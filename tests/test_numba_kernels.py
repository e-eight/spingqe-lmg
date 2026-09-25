import numpy as np
import pytest

from spingqe_lmg.core.numba_kernels import (
    _apply_x_axis,
    _apply_y_axis,
    _apply_z_axis,
    peven_sequence_energies_numba,
    statevec_sequence_energies_numba,
)
from spingqe_lmg.core.parity import build_even_table, build_h_diag
from spingqe_lmg.core.statevec_ops import (
    apply_pauli,
    hamiltonian_terms,
    initial_statevector,
    split_sv_terms,
)
from spingqe_lmg.enums import InitState
from spingqe_lmg.hamiltonians import lmg_hamiltonian
from spingqe_lmg.pools import build_collective_pool, build_pauli_pair_pool


def _random_state(n_qubits, seed):
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
    return (psi / np.linalg.norm(psi)).astype(np.complex128)


@pytest.mark.parametrize("n_qubits", [3, 4, 5])
@pytest.mark.parametrize("wire", [0, 1, 2])
def test_apply_z_axis_matches_apply_pauli(n_qubits, wire):
    if wire >= n_qubits:
        pytest.skip("wire out of range for this n_qubits")
    psi = _random_state(n_qubits, seed=n_qubits * 10 + wire)
    axis = n_qubits - 1 - wire
    got = psi.copy()
    _apply_z_axis(got, axis)
    expected = apply_pauli(psi, "Z", wire, n_qubits)
    np.testing.assert_allclose(got, expected, atol=1e-12)


@pytest.mark.parametrize("n_qubits", [3, 4, 5])
@pytest.mark.parametrize("wire", [0, 1, 2])
def test_apply_x_axis_matches_apply_pauli(n_qubits, wire):
    if wire >= n_qubits:
        pytest.skip("wire out of range for this n_qubits")
    psi = _random_state(n_qubits, seed=n_qubits * 10 + wire)
    axis = n_qubits - 1 - wire
    got = np.empty_like(psi)
    _apply_x_axis(psi, got, axis)
    expected = apply_pauli(psi, "X", wire, n_qubits)
    np.testing.assert_allclose(got, expected, atol=1e-12)


@pytest.mark.parametrize("n_qubits", [3, 4, 5])
@pytest.mark.parametrize("wire", [0, 1, 2])
def test_apply_y_axis_matches_apply_pauli(n_qubits, wire):
    if wire >= n_qubits:
        pytest.skip("wire out of range for this n_qubits")
    psi = _random_state(n_qubits, seed=n_qubits * 10 + wire)
    axis = n_qubits - 1 - wire
    got = np.empty_like(psi)
    _apply_y_axis(psi, got, axis)
    expected = apply_pauli(psi, "Y", wire, n_qubits)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_popcount64():
    from spingqe_lmg.core.numba_kernels import _popcount64

    assert _popcount64(0) == 0
    assert _popcount64(1) == 1
    assert _popcount64(0b1011) == 3
    assert _popcount64(0xFFFF) == 16


def _build_statevec_inputs(n_qubits, h, lam, pool):
    ham = lmg_hamiltonian(n_qubits, h=h, lam=lam)
    terms = hamiltonian_terms(ham)
    identity_coeff, h_diag, offdiag_data = split_sv_terms(terms, n_qubits)
    rng = np.random.default_rng(n_qubits * 7 + len(pool))
    seq = [pool[i] for i in rng.integers(0, len(pool), size=6)]
    ops_data = [
        [(word, list(wires), float(theta)) for word, wires, theta in op.factors()] for op in seq
    ]
    return ops_data, identity_coeff, h_diag, offdiag_data


@pytest.mark.parametrize("n_qubits", [4, 8, 12, 16])
@pytest.mark.parametrize("pool_kind", ["pair", "collective"])
def test_statevec_numba_matches_reference(n_qubits, pool_kind):
    from _reference_slow_statevec import reference_statevec_sequence_energies

    pool = (
        build_pauli_pair_pool(n_qubits) if pool_kind == "pair" else build_collective_pool(n_qubits)
    )
    ops_data, identity_coeff, h_diag, offdiag_terms = _build_statevec_inputs(
        n_qubits, h=1.0, lam=1.5, pool=pool
    )
    ham = lmg_hamiltonian(n_qubits, h=1.0, lam=1.5)
    terms = hamiltonian_terms(ham)
    e_ref = reference_statevec_sequence_energies(
        ops_data, terms, n_qubits, 0.0, identity_coeff, h_diag, offdiag_terms
    )
    e_numba = statevec_sequence_energies_numba(
        ops_data, n_qubits, 0.0, identity_coeff, h_diag, offdiag_terms
    )
    np.testing.assert_allclose(e_numba, e_ref, atol=1e-11)


def test_statevec_numba_empty_ops_data():
    from spingqe_lmg.core.numba_kernels import statevec_sequence_energies_numba

    result = statevec_sequence_energies_numba([], 2, 0.0, 0.0, None, [])
    assert result.shape == (0,)


def _build_peven_inputs(n_qubits, h, lam, pool):
    ham = lmg_hamiltonian(n_qubits, h=h, lam=lam)
    terms = hamiltonian_terms(ham)
    c_to_full, _full_to_c, wire0_bit = build_even_table(n_qubits)
    identity_coeff, h_diag, offdiag_terms = build_h_diag(terms, n_qubits, wire0_bit)
    # build_h_diag returns offdiag_terms with string Pauli letters (e.g. "X");
    # the Numba kernel expects ASCII-int letters, matching the conversion
    # evaluators/statevec.py's _peven_sequence_energies applies before calling
    # peven_sequence_energies_numba.
    offdiag_data = [
        (float(c), [(int(w), ord(ltr)) for w, ltr in ltrs]) for c, ltrs in offdiag_terms
    ]
    rng = np.random.default_rng(n_qubits * 11 + len(pool))
    seq = [pool[i] for i in rng.integers(0, len(pool), size=6)]
    ops_data = [
        [(word, list(wires), float(theta)) for word, wires, theta in op.factors()] for op in seq
    ]
    return ops_data, c_to_full, wire0_bit, identity_coeff, h_diag, offdiag_data


@pytest.mark.parametrize("n_qubits", [4, 8, 12, 16])
@pytest.mark.parametrize("pool_kind", ["pair", "collective"])
def test_peven_numba_matches_reference(n_qubits, pool_kind):
    from _reference_slow_statevec import reference_peven_sequence_energies

    pool = (
        build_pauli_pair_pool(n_qubits) if pool_kind == "pair" else build_collective_pool(n_qubits)
    )
    ops_data, c_to_full, wire0_bit, identity_coeff, h_diag, offdiag_data = _build_peven_inputs(
        n_qubits, h=1.0, lam=1.5, pool=pool
    )
    ham = lmg_hamiltonian(n_qubits, h=1.0, lam=1.5)
    terms = hamiltonian_terms(ham)
    # _build_peven_inputs returns offdiag_data ASCII-int-encoded (for the Numba
    # kernel); the reference implementation expects build_h_diag's raw
    # string-letter format instead, so it's recomputed here rather than reused.
    _, _, offdiag_terms = build_h_diag(terms, n_qubits, wire0_bit)
    # offdiag_terms here is string-letter format, matching build_h_diag's output.
    e_ref = reference_peven_sequence_energies(
        ops_data,
        terms,
        n_qubits,
        0.0,
        c_to_full,
        wire0_bit,
        identity_coeff,
        h_diag,
        offdiag_terms,
    )
    e_numba = peven_sequence_energies_numba(
        ops_data,
        n_qubits,
        0.0,
        c_to_full,
        wire0_bit,
        identity_coeff,
        h_diag,
        offdiag_data,  # ASCII-int format
    )
    np.testing.assert_allclose(e_numba, e_ref, atol=1e-11)


@pytest.mark.parametrize("n_qubits", [4, 6, 8])
@pytest.mark.parametrize("init_angle", [0.37, 1.1, -0.5])
def test_statevec_numba_mean_field_init(n_qubits, init_angle):
    """statevec_sequence_energies_numba with a nonzero init_angle matches
    the closed-form product state from initial_statevector, via a single
    zero-length-op no-gate sequence (energy of the bare initial state)."""
    ham = lmg_hamiltonian(n_qubits, h=1.0, lam=1.5)
    terms = hamiltonian_terms(ham)
    identity_coeff, h_diag, offdiag_terms = split_sv_terms(terms, n_qubits)

    expected_psi = initial_statevector(n_qubits, init_angle, np.complex128)
    from spingqe_lmg.core.statevec_ops import energy_from_terms

    expected_energy = energy_from_terms(expected_psi, terms, n_qubits)

    got = statevec_sequence_energies_numba(
        [[("I", [0], 0.0)]], n_qubits, init_angle, identity_coeff, h_diag, offdiag_terms
    )
    np.testing.assert_allclose(got[0], expected_energy, atol=1e-10)


@pytest.mark.parametrize("n_qubits", [4, 6, 8])
def test_statevec_numba_ghz_x_init(n_qubits):
    ham = lmg_hamiltonian(n_qubits, h=1.0, lam=1.5)
    terms = hamiltonian_terms(ham)
    identity_coeff, h_diag, offdiag_terms = split_sv_terms(terms, n_qubits)

    expected_psi = initial_statevector(n_qubits, InitState.GHZ_X.value, np.complex128)
    from spingqe_lmg.core.statevec_ops import energy_from_terms

    expected_energy = energy_from_terms(expected_psi, terms, n_qubits)

    got = statevec_sequence_energies_numba(
        [[("I", [0], 0.0)]], n_qubits, InitState.GHZ_X.value, identity_coeff, h_diag, offdiag_terms
    )
    np.testing.assert_allclose(got[0], expected_energy, atol=1e-10)


def test_statevec_numba_ghz_x_rejects_odd_n_qubits():
    with pytest.raises(ValueError, match="ghz_x"):
        statevec_sequence_energies_numba(
            [[("I", [0], 0.0)]], 5, InitState.GHZ_X.value, 0.0, None, []
        )


def test_statevec_numba_rejects_unknown_init_string():
    with pytest.raises(ValueError, match="unknown reference state"):
        statevec_sequence_energies_numba([[("I", [0], 0.0)]], 4, "not_a_real_state", 0.0, None, [])


@pytest.mark.parametrize("n_qubits", [4, 6, 8])
@pytest.mark.parametrize("init_angle", [0.37, 1.1])
def test_peven_numba_mean_field_init(n_qubits, init_angle):
    from spingqe_lmg.core.parity import build_even_table, build_h_diag
    from spingqe_lmg.core.statevec_ops import energy_from_terms

    ham = lmg_hamiltonian(n_qubits, h=1.0, lam=1.5)
    terms = hamiltonian_terms(ham)
    c_to_full, _, wire0_bit = build_even_table(n_qubits)
    identity_coeff, h_diag, offdiag_terms = build_h_diag(terms, n_qubits, wire0_bit)
    offdiag_terms_encoded = [
        (float(c), [(int(w), ord(ltr)) for w, ltr in ltrs]) for c, ltrs in offdiag_terms
    ]

    # A generic mean-field angle produces a state with both even- and
    # odd-parity components (validate_peven_preconditions rejects it for
    # production use). The Numba peven kernel, like the pre-existing
    # pure-Python fallback, reduces to the even sector via an unnormalized
    # projection (psi_full[c_to_full], no renormalization) rather than
    # rejecting it outright. Since the LMG Hamiltonian's terms are all
    # parity-preserving (no even/odd cross terms), the reference energy for
    # that unnormalized projection is obtained by zeroing the odd-sector
    # amplitudes of the full statevector before evaluating energy_from_terms
    # (comparing against the full mixed-parity state's energy would be wrong
    # here, since that includes the odd sector's own contribution too).
    expected_psi = initial_statevector(n_qubits, init_angle, np.complex128)
    even_mask = np.zeros(2**n_qubits, dtype=bool)
    even_mask[c_to_full] = True
    projected_psi = np.where(even_mask, expected_psi, 0.0)
    expected_energy = energy_from_terms(projected_psi, terms, n_qubits)

    got = peven_sequence_energies_numba(
        [[("I", [1], 0.0)]],
        n_qubits,
        init_angle,
        c_to_full,
        wire0_bit,
        identity_coeff,
        h_diag,
        offdiag_terms_encoded,
    )
    np.testing.assert_allclose(got[0], expected_energy, atol=1e-8)


@pytest.mark.parametrize("n_qubits", [4, 6, 8])
def test_peven_numba_ghz_x_init(n_qubits):
    from spingqe_lmg.core.parity import build_even_table, build_h_diag
    from spingqe_lmg.core.statevec_ops import energy_from_terms

    ham = lmg_hamiltonian(n_qubits, h=1.0, lam=1.5)
    terms = hamiltonian_terms(ham)
    c_to_full, _, wire0_bit = build_even_table(n_qubits)
    identity_coeff, h_diag, offdiag_terms = build_h_diag(terms, n_qubits, wire0_bit)
    offdiag_terms_encoded = [
        (float(c), [(int(w), ord(ltr)) for w, ltr in ltrs]) for c, ltrs in offdiag_terms
    ]

    expected_psi = initial_statevector(n_qubits, InitState.GHZ_X.value, np.complex128)
    expected_energy = energy_from_terms(expected_psi, terms, n_qubits)

    got = peven_sequence_energies_numba(
        [[("I", [1], 0.0)]],
        n_qubits,
        InitState.GHZ_X.value,
        c_to_full,
        wire0_bit,
        identity_coeff,
        h_diag,
        offdiag_terms_encoded,
    )
    np.testing.assert_allclose(got[0], expected_energy, atol=1e-8)


def test_statevec_numba_rejects_wire_out_of_range():
    offdiag_terms = [(1.0, [(5, ord("X"))])]  # wire 5, but n_qubits=4 below
    with pytest.raises(ValueError, match="wire"):
        statevec_sequence_energies_numba(
            [[("X", [0], 0.1)]], 4, 0.0, 0.0, np.zeros(16), offdiag_terms
        )


def test_peven_numba_rejects_wire_out_of_range():
    c_to_full, _, wire0_bit = build_even_table(4)
    offdiag_terms = [(1.0, [(5, ord("X"))])]
    with pytest.raises(ValueError, match="wire"):
        peven_sequence_energies_numba(
            [[("X", [1], 0.1)]], 4, 0.0, c_to_full, wire0_bit, 0.0, np.zeros(8), offdiag_terms
        )


def test_flatten_ops_matches_manual_flattening():
    from spingqe_lmg.core.numba_kernels import _flatten_ops

    ops_data = [[("XY", [0, 1], 0.3)], [("Z", [2], 0.1), ("X", [0], 0.2)]]
    op_offsets, factor_offsets, factor_letters, factor_wires, factor_thetas = _flatten_ops(ops_data)
    assert list(op_offsets) == [0, 1, 3]
    assert list(factor_offsets) == [0, 2, 3, 4]
    assert list(factor_letters) == [ord("X"), ord("Y"), ord("Z"), ord("X")]
    assert list(factor_wires) == [0, 1, 2, 0]
    assert list(factor_thetas) == [0.3, 0.1, 0.2]


def test_flatten_offdiag_terms_matches_manual_flattening():
    from spingqe_lmg.core.numba_kernels import _flatten_offdiag_terms

    offdiag_terms = [(1.5, [(0, ord("X")), (1, ord("Y"))]), (2.0, [(2, ord("Z"))])]
    term_offsets, term_wires, term_letters, coeffs = _flatten_offdiag_terms(offdiag_terms)
    assert list(term_offsets) == [0, 2, 3]
    assert list(term_wires) == [0, 1, 2]
    assert list(term_letters) == [ord("X"), ord("Y"), ord("Z")]
    assert list(coeffs) == [1.5, 2.0]
