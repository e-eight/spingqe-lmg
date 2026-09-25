import numpy as np
import pennylane as qml
import pytest
from _reference_slow_statevec import _apply_word_peven

from spingqe_lmg.config import config_from_dict
from spingqe_lmg.core.parity import (
    build_even_table,
    build_h_diag,
    build_offdiag_perms,
    pauli_word_parity,
    pool_parity_summary,
)
from spingqe_lmg.core.statevec_ops import (
    apply_jx,
    apply_jy,
    apply_pauli,
    energy_from_terms,
    hamiltonian_terms,
    initial_statevector,
    order_param_sq_jx,
)
from spingqe_lmg.evaluators.pennylane import PennyLaneEvaluator, initial_energy
from spingqe_lmg.evaluators.statevec import IncrementalEvaluator, ParityEvenEvaluator
from spingqe_lmg.exact import dense_matrix, dicke_matrix, ground_state
from spingqe_lmg.hamiltonians import lmg_hamiltonian, lmg_mean_field_angle
from spingqe_lmg.pools import build_collective_pool, build_pauli_pair_pool
from spingqe_lmg.training import train

PAULI_OPS = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
EXT_MF_PAULIS = ("ZZ", "XX", "YY", "YZ", "XY")


def random_state(n_qubits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
    return psi / np.linalg.norm(psi)


@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
def test_apply_pauli_matches_dense(n, pauli):
    psi = random_state(n, seed=n)
    for wire in range(n):
        dense = dense_matrix(qml.Hamiltonian([1.0], [PAULI_OPS[pauli](wire)]), n)
        np.testing.assert_allclose(apply_pauli(psi, pauli, wire, n), dense @ psi, atol=1e-12)


@pytest.mark.parametrize("apply_j,pauli", [(apply_jx, qml.PauliX), (apply_jy, qml.PauliY)])
def test_apply_collective_matches_dense(apply_j, pauli):
    n = 4
    psi = random_state(n, seed=17)
    j_dense = 0.5 * sum(dense_matrix(qml.Hamiltonian([1.0], [pauli(i)]), n) for i in range(n))
    np.testing.assert_allclose(apply_j(psi, n), j_dense @ psi, atol=1e-12)


def test_apply_pauli_rejects_unknown():
    with pytest.raises(ValueError, match="unknown Pauli"):
        apply_pauli(random_state(2, seed=0), "Q", 0, 2)


def test_order_param_sq_jx_matches_dense():
    n = 4
    psi = random_state(n, seed=23)
    jx = 0.5 * sum(dense_matrix(qml.Hamiltonian([1.0], [qml.PauliX(i)]), n) for i in range(n))
    expected = float(np.real(psi.conj() @ (jx @ jx) @ psi)) * 4 / n**2
    assert order_param_sq_jx(psi, n) == pytest.approx(expected, abs=1e-12)


# --- incremental prefix evaluator (item e) -------------------------------


def _init_angle(kind, h, lam):
    if kind == "zero":
        return 0.0
    if kind == "mean_field":
        return lmg_mean_field_angle(h, lam)
    return "ghz_x"


@pytest.mark.parametrize("n", [4, 6, 8])
@pytest.mark.parametrize("init", ["zero", "mean_field", "ghz_x"])
@pytest.mark.parametrize("gamma", [0.0, 0.5])
def test_statevec_matches_snapshot(n, init, gamma):
    # the acceptance gate: the incremental stepper must reproduce the snapshot
    # evaluator's per-prefix energies end to end (init + stepping + energy)
    h, lam = 1.0, 1.5
    ham = lmg_hamiltonian(n, h=h, lam=lam, gamma=gamma)
    pool = build_pauli_pair_pool(n, paulis=EXT_MF_PAULIS)
    angle = _init_angle(init, h, lam)
    idx = np.random.default_rng(3).integers(0, len(pool), size=(4, 10))
    ref = PennyLaneEvaluator(pool, ham, n, init_angle=angle)(idx)
    sv = IncrementalEvaluator(pool, ham, n, init_angle=angle)(idx)
    np.testing.assert_allclose(sv, ref, atol=1e-9)


def test_statevec_initial_energy_matches():
    n, h, lam = 6, 1.0, 1.8
    theta = lmg_mean_field_angle(h, lam)
    ham = lmg_hamiltonian(n, h=h, lam=lam)
    ev = IncrementalEvaluator(build_pauli_pair_pool(n), ham, n, init_angle=theta)
    assert ev.initial_energy() == pytest.approx(initial_energy(ham, n, theta), abs=1e-9)


def test_statevec_njobs_consistent():
    n = 6
    ham = lmg_hamiltonian(n, h=1.0, lam=1.5)
    pool = build_pauli_pair_pool(n)
    idx = np.random.default_rng(2).integers(0, len(pool), size=(8, 10))
    serial = IncrementalEvaluator(pool, ham, n, n_jobs=1)(idx)
    parallel = IncrementalEvaluator(pool, ham, n, n_jobs=4)(idx)
    np.testing.assert_allclose(serial, parallel, atol=1e-12)


def test_statevec_complex64_warns_and_computes_in_complex128():
    """dtype=complex64 is no longer natively supported; Numba always computes
    in complex128. A RuntimeWarning is issued, and results match a genuine
    complex128 run exactly (same underlying computation, not just close)."""
    n = 6
    ham = lmg_hamiltonian(n, h=1.0, lam=2.0)
    pool = build_pauli_pair_pool(n, paulis=EXT_MF_PAULIS)
    idx = np.random.default_rng(1).integers(0, len(pool), size=(8, 12))
    e128 = IncrementalEvaluator(pool, ham, n, dtype=np.complex128)(idx)
    with pytest.warns(RuntimeWarning, match="complex64"):
        ev64 = IncrementalEvaluator(pool, ham, n, dtype=np.complex64)
    e64 = ev64(idx)
    np.testing.assert_array_equal(e64, e128)


# --- change C: buffer reuse in energy_from_terms -------------------------


def test_energy_from_terms_buffer_matches_allocating():
    """energy_from_terms with pre-allocated buffers == allocating path."""
    n = 6
    ham = lmg_hamiltonian(n, h=1.0, lam=1.8)
    psi = random_state(n, seed=42)
    terms = hamiltonian_terms(ham)
    buf_a = np.empty(2**n, dtype=psi.dtype)
    buf_b = np.empty(2**n, dtype=psi.dtype)
    e_alloc = energy_from_terms(psi, terms, n)
    e_buf = energy_from_terms(psi, terms, n, buf_a=buf_a, buf_b=buf_b)
    assert e_buf == pytest.approx(e_alloc, abs=1e-12)


# --- parity utilities ----------------------------------------------------


def test_pauli_word_parity_classification():
    assert pauli_word_parity("ZZ") == 0  # even
    assert pauli_word_parity("XX") == 0  # even
    assert pauli_word_parity("YY") == 0  # even
    assert pauli_word_parity("XY") == 0  # even (2 X/Y letters)
    assert pauli_word_parity("Z") == 0  # even
    assert pauli_word_parity("YZ") == 1  # odd
    assert pauli_word_parity("ZX") == 1  # odd
    assert pauli_word_parity("X") == 1  # odd


def test_pool_parity_summary_default_pool():
    """Default pauli_pair pool (ZZ/XX/YY + single-Z) is entirely parity-even."""
    n = 4
    pool = build_pauli_pair_pool(n)  # default paulis=("ZZ", "XX", "YY")
    summary = pool_parity_summary(pool)
    assert all(v == "even" for v in summary.values()), f"expected all even, got: {summary}"


def test_pool_parity_summary_extended_pool():
    """Extended pool with YZ contains parity-odd operators."""
    n = 4
    pool = build_pauli_pair_pool(n, paulis=("ZZ", "XX", "YY", "YZ", "XY"))
    summary = pool_parity_summary(pool)
    assert summary.get("YZ") == "odd"
    assert summary.get("XX") == "even"
    assert summary.get("XY") == "even"


# --- change E: LRU cache cap ---------------------------------------------


def test_statevec_cache_maxsize():
    """Verify LRU eviction: filling past cache_maxsize evicts the oldest entry."""
    from unittest.mock import patch

    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=1.5)
    pool = build_pauli_pair_pool(n)
    ev = IncrementalEvaluator(pool, ham, n, cache_maxsize=2)
    seq_a = np.array([[0]])
    seq_b = np.array([[1]])
    seq_c = np.array([[0, 1]])
    _ = ev(seq_a)
    _ = ev(seq_b)
    _ = ev(seq_c)
    with patch.object(ev, "_sequence_energies", wraps=ev._sequence_energies) as mock_fn:
        _ = ev(seq_a)
        assert mock_fn.call_count == 1, "seq_a should be a cache miss after eviction"


def test_micro_train_with_statevec_evaluator(tmp_path):
    cfg = config_from_dict(
        {
            "run_name": "statevec-micro",
            "hamiltonian": {"kind": "lmg", "n_qubits": 6, "lam": 1.5},
            "pool": {"kind": "pauli_pair", "connectivity": "all"},
            "model": {"n_layer": 1, "n_head": 2, "n_embd": 16, "dropout": 0.0},
            "train": {
                "epochs": 2,
                "seq_gen": 4,
                "seq_len": 4,
                "n_batches": 2,
                "eval_iter": 2,
                "eval_sequences": 4,
                "device": "cpu",
                "evaluator": "incremental",
            },
        }
    )
    result = train(cfg, tmp_path / "run")
    e0 = ground_state(dicke_matrix(6, 1.0, 1.5))[0]
    assert result.ground_energy == pytest.approx(e0)
    assert result.best_energy >= e0 - 1e-9
    assert result.refined_energy is not None


def test_lru_cache_get_updates_recency():
    """LRUCache.get() must move the accessed key to the end (most recent)."""
    from spingqe_lmg.core.caching import LRUCache

    cache = LRUCache(maxsize=3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    assert next(iter(cache)) == "a"
    cache.get("a")
    assert next(iter(cache)) == "b", "get('a') should move 'a' to end, 'b' should be oldest"
    cache["d"] = 4
    assert "b" not in cache
    assert "a" in cache


def test_lru_cache_pickles_and_deepcopies():
    """LRUCache must survive pickle/deepcopy, preserving maxsize and eviction."""
    import copy
    import pickle

    from spingqe_lmg.core.caching import LRUCache

    cache = LRUCache(maxsize=2)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3  # evicts "a"

    restored = pickle.loads(pickle.dumps(cache))
    assert restored.maxsize == 2
    assert dict(restored) == {"b": 2, "c": 3}
    restored["d"] = 4  # eviction still active post-restore
    assert "b" not in restored

    cloned = copy.deepcopy(cache)
    assert cloned.maxsize == 2
    assert dict(cloned) == {"b": 2, "c": 3}


# --- F-04: apply_pauli_rot buffer aliasing guard -----------------------


def test_apply_pauli_rot_rejects_aliased_out_buf():
    """apply_pauli_rot must raise ValueError when out aliases a scratch buffer."""
    import numpy as np
    import pytest
    from _reference_slow_statevec import apply_pauli_rot

    n = 4
    psi = np.ones(2**n, dtype=complex) / np.sqrt(2**n)
    buf = np.zeros(2**n, dtype=complex)

    word = ["X", "I", "I", "I"]
    wires = [0, 1, 2, 3]
    theta = 0.1

    # Passing buf as both out and buf_a must raise
    with pytest.raises(ValueError, match="alias"):
        apply_pauli_rot(psi, word, wires, n, theta, out=buf, buf_a=buf)


# ---------------------------------------------------------------------------
# Sector-table and wire0_bit correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 4])
def test_c_to_full_all_even_parity(n):
    c_to_full, _, _ = build_even_table(n)
    parities = np.array([bin(int(x)).count("1") % 2 for x in c_to_full])
    assert np.all(parities == 0)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_c_to_full_correct_count(n):
    c_to_full, _, _ = build_even_table(n)
    assert len(c_to_full) == 2 ** (n - 1)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_wire0_bit_matches_popcount_parity(n):
    _, _, wire0_bit = build_even_table(n)
    half = 2 ** (n - 1)
    for k in range(half):
        expected = bin(k).count("1") % 2
        assert bool(wire0_bit[k]) == bool(expected), f"k={k}"


def test_build_even_table_is_cached():
    """Same object returned for the same n_qubits (lru_cache active)."""
    result_a = build_even_table(4)
    result_b = build_even_table(4)
    assert result_a[0] is result_b[0]
    assert result_a[2] is result_b[2]


# ---------------------------------------------------------------------------
# Axis-manipulation correctness (unified path for all Pauli letters)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,wire,pauli",
    [
        (2, 0, "X"),
        (2, 1, "X"),
        (2, 0, "Y"),
        (2, 1, "Y"),
        (2, 0, "Z"),
        (2, 1, "Z"),
        (3, 0, "X"),
        (3, 1, "Y"),
        (3, 2, "Z"),
    ],
)
def test_single_pauli_on_the_fly_matches_apply_pauli(n, wire, pauli):
    """Compressed-space axis-manipulation matches full-space apply_pauli."""
    if pauli_word_parity(pauli) != 0:
        pytest.skip(f"single {pauli!r} is parity-odd; PEven does not support it")

    c_to_full, _, wire0_bit = build_even_table(n)
    half = len(c_to_full)

    rng = np.random.default_rng(n * 10 + wire)
    psi_full = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    odd_mask = np.ones(2**n, dtype=bool)
    odd_mask[c_to_full] = False
    psi_full[odd_mask] = 0.0
    psi_full /= np.linalg.norm(psi_full)

    psi_c = psi_full[c_to_full]

    # Full-space reference
    ref_full = apply_pauli(psi_full, pauli, wire, n)

    # Axis-manipulation in compressed space
    out = np.empty(half, dtype=psi_c.dtype)
    buf = np.empty(half, dtype=psi_c.dtype)
    wire0_buf = np.empty(half, dtype=psi_c.dtype)
    _apply_word_peven(
        psi_c,
        pauli,
        (wire,),
        n,
        wire0_bit,
        out=out,
        buf=buf,
        dtype_name=psi_c.dtype.name,
        wire0_buf=wire0_buf,
    )
    result_full_from_c = np.zeros(2**n, dtype=complex)
    result_full_from_c[c_to_full] = out

    np.testing.assert_allclose(result_full_from_c, ref_full, atol=1e-12)


@pytest.mark.parametrize(
    "n,word,wires",
    [
        (3, "XX", (0, 1)),
        (3, "YY", (1, 2)),
        (3, "ZZ", (0, 2)),
        (3, "XY", (0, 1)),
        (4, "XX", (0, 3)),
        (3, "ZZZ", (0, 1, 2)),
    ],
)
def test_two_qubit_on_the_fly_matches_full_space(n, word, wires):
    """Multi-qubit even-parity application matches sequential apply_pauli."""
    c_to_full, _, wire0_bit = build_even_table(n)
    half = len(c_to_full)

    rng = np.random.default_rng(hash((n, word, wires)) % (2**31))
    psi_full = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    odd_mask = np.ones(2**n, dtype=bool)
    odd_mask[c_to_full] = False
    psi_full[odd_mask] = 0.0
    psi_full /= np.linalg.norm(psi_full)

    psi_c = psi_full[c_to_full]

    # Full-space reference via sequential apply_pauli
    ref_full = psi_full.copy()
    for letter, w in zip(word, wires, strict=True):
        ref_full = apply_pauli(ref_full, letter, w, n)

    # Axis-manipulation in compressed space
    out = np.empty(half, dtype=psi_c.dtype)
    buf = np.empty(half, dtype=psi_c.dtype)
    wire0_buf = np.empty(half, dtype=psi_c.dtype)
    _apply_word_peven(
        psi_c,
        word,
        wires,
        n,
        wire0_bit,
        out=out,
        buf=buf,
        dtype_name=psi_c.dtype.name,
        wire0_buf=wire0_buf,
    )
    result_full_from_c = np.zeros(2**n, dtype=complex)
    result_full_from_c[c_to_full] = out

    np.testing.assert_allclose(result_full_from_c, ref_full, atol=1e-12)


# ---------------------------------------------------------------------------
# PEven evaluator vs IncrementalEvaluator (end-to-end)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [4, 6])
@pytest.mark.parametrize("lam", [0.8, 1.5])
def test_peven_matches_statevec_default_pool(n, lam):
    """ParityEvenEvaluator agrees with IncrementalEvaluator (default pool)."""
    ham = lmg_hamiltonian(n, h=1.0, lam=lam)
    pool = build_pauli_pair_pool(n)
    rng = np.random.default_rng(n + int(lam * 10))
    batch = rng.integers(0, len(pool), size=(4, 8))

    ref = IncrementalEvaluator(pool, ham, n)(batch)
    pev = ParityEvenEvaluator(pool, ham, n)(batch)
    np.testing.assert_allclose(pev, ref, atol=1e-10, err_msg=f"mismatch at N={n}, lam={lam}")


def test_peven_rejects_parity_odd_pool():
    """ParityEvenEvaluator raises for pools containing parity-odd ops (YZ)."""
    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=1.5)
    pool = build_pauli_pair_pool(n, paulis=("ZZ", "XX", "YY", "YZ", "XY"))
    with pytest.raises(ValueError, match="parity-odd"):
        ParityEvenEvaluator(pool, ham, n)


def test_peven_initial_energy_matches_statevec():
    n, lam = 6, 1.5
    ham = lmg_hamiltonian(n, h=1.0, lam=lam)
    pool = build_pauli_pair_pool(n)
    sv = IncrementalEvaluator(pool, ham, n)
    pev = ParityEvenEvaluator(pool, ham, n)
    assert pev.initial_energy() == pytest.approx(sv.initial_energy(), abs=1e-10)


def test_peven_ghz_x_initial_state():
    """ghz_x is a valid parity-even initial state for PEven."""
    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=1.5)
    pool = build_pauli_pair_pool(n)
    rng = np.random.default_rng(0)
    batch = rng.integers(0, len(pool), size=(2, 4))

    ref = IncrementalEvaluator(pool, ham, n, init_angle="ghz_x")(batch)
    pev = ParityEvenEvaluator(pool, ham, n, init_angle="ghz_x")(batch)
    np.testing.assert_allclose(pev, ref, atol=1e-10)


@pytest.mark.parametrize("n", [4, 6])
@pytest.mark.parametrize("lam", [0.8, 2.0])
def test_peven_matches_statevec_collective_pool(n, lam):
    """PEven agrees with statevec evaluator for the collective pool."""
    ham = lmg_hamiltonian(n, h=1.0, lam=lam)
    pool = build_collective_pool(n)
    rng = np.random.default_rng(n * 100 + int(lam * 10))
    batch = rng.integers(0, len(pool), size=(3, 6))

    ref = IncrementalEvaluator(pool, ham, n)(batch)
    pev = ParityEvenEvaluator(pool, ham, n)(batch)
    np.testing.assert_allclose(
        pev, ref, atol=1e-10, err_msg=f"collective pool mismatch at N={n}, lam={lam}"
    )


def test_peven_n_jobs_parallel_matches_serial():
    """n_jobs=2 parallel path produces the same energies as serial (n_jobs=1)."""
    n, lam = 6, 1.5
    ham = lmg_hamiltonian(n, h=1.0, lam=lam)
    pool = build_pauli_pair_pool(n)
    rng = np.random.default_rng(77)
    batch = rng.integers(0, len(pool), size=(4, 8))

    serial = ParityEvenEvaluator(pool, ham, n, n_jobs=1)(batch)
    parallel = ParityEvenEvaluator(pool, ham, n, n_jobs=2)(batch)
    np.testing.assert_allclose(
        parallel, serial, atol=1e-12, err_msg="parallel n_jobs=2 differs from serial"
    )


# ---------------------------------------------------------------------------
# Rejection
# ---------------------------------------------------------------------------


def test_peven_rejects_nonzero_float_init_angle():
    """Mean-field product state spans both sectors — should raise ValueError."""
    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=2.0)
    pool = build_pauli_pair_pool(n)
    theta = lmg_mean_field_angle(1.0, 2.0)
    assert theta > 0.1
    with pytest.raises(ValueError, match="odd-parity components"):
        ParityEvenEvaluator(pool, ham, n, init_angle=theta)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def test_peven_compressed_state_is_half_size():
    for n in [4, 6, 8]:
        c_to_full, _, _ = build_even_table(n)
        assert len(c_to_full) == 2 ** (n - 1)


def test_peven_no_kernel_storage():
    """No precomputed kernels stored — only index tables."""
    n = 6
    ham = lmg_hamiltonian(n, h=1.0, lam=1.5)
    pool = build_pauli_pair_pool(n)
    pev = ParityEvenEvaluator(pool, ham, n)
    assert not hasattr(pev, "_kernels")
    rng = np.random.default_rng(42)
    batch = rng.integers(0, len(pool), size=(2, 4))
    ref = IncrementalEvaluator(pool, ham, n)(batch)
    np.testing.assert_allclose(pev(batch), ref, atol=1e-10)


def test_build_even_table_wire0_bit():
    _, _, wire0_bit = build_even_table(4)
    assert not wire0_bit[0]
    assert wire0_bit[1]
    assert wire0_bit[2]
    assert not wire0_bit[3]


# ---------------------------------------------------------------------------
# Fix C: h_diag precomputation correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,lam,gamma",
    [
        (4, 0.8, 0.0),
        (4, 1.5, 0.0),
        (4, 1.5, 1.0),
        (6, 1.5, 0.0),
    ],
)
def test_h_diag_matches_per_term_energy(n, lam, gamma):
    """h_diag + offdiag_terms reproduces the full Hamiltonian energy at N=4,6."""
    from _reference_slow_statevec import _energy_from_terms_peven

    ham = lmg_hamiltonian(n, h=1.0, lam=lam, gamma=gamma)
    _, _, wire0_bit = build_even_table(n)
    terms = hamiltonian_terms(ham)

    identity_coeff, h_diag, offdiag_terms = build_h_diag(terms, n, wire0_bit)

    # identity_coeff + h_diag + offdiag_terms should account for all terms
    assert len(offdiag_terms) + sum(1 for _c, letters in terms if not letters) + len(h_diag) > 0

    half = 1 << (n - 1)
    rng = np.random.default_rng(n * 100 + int(lam * 10))
    psi = rng.normal(size=half) + 1j * rng.normal(size=half)
    psi /= np.linalg.norm(psi)

    scratch = np.empty(half, dtype=psi.dtype)
    buf = np.empty(half, dtype=psi.dtype)
    wire0_buf = np.empty(half, dtype=psi.dtype)

    # Reference: full term-by-term evaluation
    e_ref = _energy_from_terms_peven(
        psi,
        terms,
        n,
        wire0_bit,
        scratch,
        buf,
        psi.dtype.name,
        wire0_buf,
    )

    # Fast path: diagonal dot + off-diagonal loop
    prob = psi.real**2 + psi.imag**2
    e_fast = identity_coeff + np.dot(h_diag, prob)
    e_fast += _energy_from_terms_peven(
        psi,
        offdiag_terms,
        n,
        wire0_bit,
        scratch,
        buf,
        psi.dtype.name,
        wire0_buf,
    )

    assert e_fast == pytest.approx(e_ref, abs=1e-10), (
        f"h_diag energy mismatch at N={n}, lam={lam}, gamma={gamma}: "
        f"fast={e_fast:.12f} ref={e_ref:.12f}"
    )


def test_h_diag_is_readonly():
    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=1.5)
    _, _, wire0_bit = build_even_table(n)

    _, h_diag, _ = build_h_diag(hamiltonian_terms(ham), n, wire0_bit)
    assert not h_diag.flags.writeable


# ---------------------------------------------------------------------------
# Fix D: off-diagonal permutation correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,lam,gamma",
    [
        (4, 0.8, 0.0),
        (4, 1.5, 0.0),
        (4, 1.5, 1.0),
        (6, 1.5, 0.0),
    ],
)
def test_offdiag_perms_match_apply_word(n, lam, gamma):
    """Off-diagonal perm+phase matches _apply_word_peven for each term."""
    ham = lmg_hamiltonian(n, h=1.0, lam=lam, gamma=gamma)
    _, _, wire0_bit = build_even_table(n)
    terms = hamiltonian_terms(ham)
    _, _, offdiag_terms = build_h_diag(terms, n, wire0_bit)

    result = build_offdiag_perms(offdiag_terms, n, wire0_bit)
    if result is None:
        pytest.skip("N too large for precomputed perms (memory guard)")
    perms, phases, coeffs = result

    half = 1 << (n - 1)
    rng = np.random.default_rng(n * 200 + int(lam * 10))
    psi = rng.normal(size=half) + 1j * rng.normal(size=half)
    psi /= np.linalg.norm(psi)
    psi_conj = np.conj(psi)

    scratch = np.empty(half, dtype=psi.dtype)
    buf = np.empty(half, dtype=psi.dtype)
    wire0_buf = np.empty(half, dtype=psi.dtype)

    for idx, (_coeff, letters) in enumerate(offdiag_terms):
        word = "".join(ltr for _w, ltr in letters)
        wires = tuple(w for w, _ltr in letters)

        # Reference via _apply_word_peven
        phi_ref = _apply_word_peven(
            psi,
            word,
            wires,
            n,
            wire0_bit,
            out=scratch,
            buf=buf,
            dtype_name=psi.dtype.name,
            wire0_buf=wire0_buf,
        )
        e_ref = np.dot(psi_conj, phi_ref).real

        # Precomputed path
        if phases is not None:
            phi_fast = phases[idx] * psi[perms[idx]]
        else:
            phi_fast = psi[perms[idx]]
        e_fast = np.dot(psi_conj, phi_fast).real

        assert e_fast == pytest.approx(e_ref, abs=1e-10), (
            f"perm mismatch for term {idx} ({word} on {wires}) at N={n}: "
            f"fast={e_fast:.12f} ref={e_ref:.12f}"
        )


@pytest.mark.parametrize(
    "n,lam,gamma",
    [
        (4, 0.8, 0.0),
        (4, 1.5, 1.0),
        (6, 1.5, 0.0),
    ],
)
def test_full_fast_path_matches_statevec(n, lam, gamma):
    """Full fast energy path (h_diag + perms) matches IncrementalEvaluator."""
    ham = lmg_hamiltonian(n, h=1.0, lam=lam, gamma=gamma)
    pool = build_pauli_pair_pool(n)
    rng = np.random.default_rng(n * 300 + int(lam * 10))
    batch = rng.integers(0, len(pool), size=(3, 6))

    ref = IncrementalEvaluator(pool, ham, n)(batch)
    pev = ParityEvenEvaluator(pool, ham, n)(batch)
    np.testing.assert_allclose(
        pev, ref, atol=1e-10, err_msg=f"fast-path mismatch N={n}, lam={lam}, gamma={gamma}"
    )


# ---------------------------------------------------------------------------
# Non-uniform Hamiltonian: peven must match full-basis ground truth
# ---------------------------------------------------------------------------


def test_peven_matches_incremental_nonuniform_field():
    """ParityEvenEvaluator must agree with the full-basis ground truth under a
    non-uniform per-site Z field, not just the uniform LMG field.

    Originally written to catch a wire/bit convention bug in the (now-removed)
    Rust peven.rs kernel: because the standard LMG Hamiltonian has a *uniform*
    Z-field coefficient across all wires, a wire-numbering bug there would sum
    the same value over the same complete set of bit positions regardless of
    convention -- so every test using only lmg_hamiltonian would pass either
    way. A non-uniform per-site field breaks that symmetry. Kept as a general
    correctness guard on the Numba peven kernel's wire/bit convention.
    """
    import pennylane as qml

    n = 6
    rng = np.random.default_rng(0)
    h_i = rng.normal(size=n)
    coeffs = [float(h_i[i]) for i in range(n)]
    ops = [qml.PauliZ(i) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            coeffs.append(-0.3 / n)
            ops.append(qml.PauliX(i) @ qml.PauliX(j))
    ham = qml.Hamiltonian(coeffs, ops)

    pool = build_pauli_pair_pool(n)
    batch = rng.integers(0, len(pool), size=(3, 5))

    ref = IncrementalEvaluator(pool, ham, n)(batch)
    peven_result = ParityEvenEvaluator(pool, ham, n)(batch)

    np.testing.assert_allclose(
        peven_result,
        ref,
        atol=1e-10,
        err_msg="peven disagrees with full-basis ground truth under a non-uniform Z field",
    )


def test_peven_matches_incremental_nonuniform_coupling():
    """ParityEvenEvaluator must agree with the full-basis ground truth under a
    non-uniform XX coupling, not just uniform LMG coupling.

    See test_peven_matches_incremental_nonuniform_field -- same rationale,
    for the off-diagonal (coupling) terms instead of the diagonal (field) term.
    """
    import pennylane as qml

    n = 6
    rng = np.random.default_rng(1)
    coeffs = [-0.5 for _ in range(n)]
    ops = [qml.PauliZ(i) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            coeffs.append(float(rng.normal()))
            ops.append(qml.PauliX(i) @ qml.PauliX(j))
    ham = qml.Hamiltonian(coeffs, ops)

    pool = build_pauli_pair_pool(n)
    batch = rng.integers(0, len(pool), size=(3, 5))

    ref = IncrementalEvaluator(pool, ham, n)(batch)
    peven_result = ParityEvenEvaluator(pool, ham, n)(batch)

    np.testing.assert_allclose(
        peven_result,
        ref,
        atol=1e-10,
        err_msg="peven disagrees with full-basis ground truth under a non-uniform XX coupling",
    )


# ---------------------------------------------------------------------------
# Regression: build_even_table must use bounded @lru_cache, not @cache (#54)
# ---------------------------------------------------------------------------


def test_build_even_table_small_cache():
    """build_even_table must use @lru_cache, not unbounded @cache.

    @cache is a thin wrapper around @lru_cache(maxsize=None), so both
    expose ``cache_info()``.  The real fix is a bounded maxsize."""
    info = build_even_table.cache_info()
    assert info.maxsize is not None, (
        f"maxsize={info.maxsize}; build_even_table must use "
        f"@lru_cache(maxsize=...) not unbounded @cache"
    )
    assert info.maxsize > 0, f"maxsize={info.maxsize} must be positive"


# ---------------------------------------------------------------------------
# P4: Pre-allocate phi buffer in _peven_sequence_energies
# ---------------------------------------------------------------------------


def test_energy_fast_peven_reuses_phi_buffer():
    """_energy_fast_peven must accept a pre-allocated phi parameter."""
    from _reference_slow_statevec import _energy_fast_peven

    from spingqe_lmg.hamiltonians import lmg_hamiltonian

    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=2.0)
    terms = hamiltonian_terms(ham)
    c_to_full, _, wire0_bit = build_even_table(n)
    half = 1 << (n - 1)
    id_c, h_diag, offdiag_terms = build_h_diag(terms, n, wire0_bit)
    result = build_offdiag_perms(offdiag_terms, n, wire0_bit)
    od_perms, od_phases, od_coeffs = result if result else (None, None, None)
    psi_full = initial_statevector(n, 0.0, np.dtype(np.complex128))
    psi = psi_full[c_to_full]
    scratch = np.empty(half, dtype=np.complex128)
    buf = np.empty(half, dtype=np.complex128)
    wire0_buf = np.empty(half, dtype=np.complex128)
    phi = np.empty(half, dtype=np.complex128)

    # Should accept phi param without TypeError
    e = _energy_fast_peven(
        psi,
        id_c,
        h_diag,
        offdiag_terms,
        n,
        wire0_bit,
        scratch,
        buf,
        "complex128",
        wire0_buf,
        od_perms,
        od_phases,
        od_coeffs,
        phi=phi,
    )
    assert np.isfinite(e)
