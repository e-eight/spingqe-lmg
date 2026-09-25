"""Tests for IncrementalGPUEvaluator and ParityEvenGPUEvaluator (statevec_gpu.py).

Correctness for the parity-even class is verified against ParityEvenEvaluator
at small N. All non-GPU-marked tests run on device="cpu" so they pass without
a CUDA device. GPU-specific tests require pytest -m gpu.
"""

import numpy as np
import pytest

from spingqe_lmg.evaluators.statevec import ParityEvenEvaluator
from spingqe_lmg.evaluators.statevec_gpu import (
    IncrementalGPUEvaluator,
    ParityEvenGPUEvaluator,
)
from spingqe_lmg.evaluators.statevec_gpu import (
    _precompute_gate_factor_peven as _precompute_gate_factor,
)
from spingqe_lmg.hamiltonians import lmg_hamiltonian, lmg_mean_field_angle
from spingqe_lmg.pools import build_collective_pool, build_pauli_pair_pool

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")

# All correctness tests run on CUDA device to avoid requiring a GPU.
CUDA = "cuda"
ATOL_FP32 = 2e-5  # FP32 vs FP64 tolerance
ATOL_FP64 = 1e-10  # FP64 vs FP64 tolerance


# ---------------------------------------------------------------------------
# IncrementalGPUEvaluator (full Hilbert space) -- moved from test_statevec.py
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_statevec_gpu_parity_cache():
    """Parity cache avoids recomputing parity_np on every energy eval."""
    from unittest.mock import patch

    from spingqe_lmg.core.torch_utils import parity_np

    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=2.0)
    pool = build_pauli_pair_pool(n)
    ev = IncrementalGPUEvaluator(pool, ham, n, dtype=torch.complex64, device="cuda")
    idx = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int64)
    with patch("spingqe_lmg.core.torch_utils.parity_np", wraps=parity_np) as mock_pn:
        ev(idx)
        mock_pn.reset_mock()
        ev(idx)
    assert mock_pn.call_count == 0, (
        f"cached parity should not call parity_np again, got {mock_pn.call_count}"
    )


def test_statevec_gpu_gate_factor_caches_sign_float():
    """sign_float (not just the raw int8 parity tensor) must be cached per sign_mask.

    Mirrors test_gate_factor_peven_caches_sign_float below for the
    full-Hilbert _precompute_gate_factor (IncrementalGPUEvaluator's
    precompute path) -- same bug, same fix, both functions in
    statevec_gpu.py share the identical pattern.
    """
    from spingqe_lmg.evaluators.statevec_gpu import _precompute_gate_factor as _pgf_full

    cache: dict = {}
    n = 6
    n2 = 1 << n
    device = torch.device("cpu")

    _, sign_a, _, _, _, _ = _pgf_full("ZZ", (1, 2), 0.0, n, n2, device, cache=cache)
    _, sign_b, _, _, _, _ = _pgf_full("ZZ", (1, 2), 0.7, n, n2, device, cache=cache)

    assert sign_a is not None
    assert sign_a is sign_b, "sign_float tensor was reallocated on a cache hit"


def test_incremental_gpu_arange_cached():
    """IncrementalGPUEvaluator must expose _arange matching state-vector size."""
    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=1.5)
    pool = build_pauli_pair_pool(n)
    ev = IncrementalGPUEvaluator(pool, ham, n, dtype=torch.complex64, device="cpu")
    assert hasattr(ev, "_arange"), "IncrementalGPUEvaluator must cache _arange"
    assert len(ev._arange) == (1 << n), f"_arange length should be 2^n={1 << n}"
    assert ev._arange.dtype == torch.int64


# ---------------------------------------------------------------------------
# _precompute_gate_factor_peven unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,word,wires",
    [
        (4, "ZZ", (0, 1)),
        (4, "XX", (0, 1)),
        (4, "YY", (0, 1)),
        (4, "ZZ", (1, 2)),
        (6, "XX", (0, 3)),
        (6, "YY", (2, 4)),
    ],
)
def test_gate_factor_phase_matches_apply_word_peven(n, word, wires):
    """XOR-gather with precomputed phase matches _apply_word_peven for Z, X, Y words."""
    from _reference_slow_statevec import _apply_word_peven

    from spingqe_lmg.core.parity import build_even_table

    c_to_full, _, wire0_bit = build_even_table(n)
    half = 1 << (n - 1)
    rng = np.random.default_rng(42)
    psi_np = (rng.random(half) + 1j * rng.random(half)).astype(np.complex128)
    psi_np /= np.linalg.norm(psi_np)

    xor_mask, parity_tensor, c_phase, _, _ = _precompute_gate_factor(
        word, wires, 0.0, n, half, torch.device("cpu")
    )

    # Reference: _apply_word_peven
    scratch = np.empty(half, dtype=np.complex128)
    out = np.empty(half, dtype=np.complex128)
    wire0_buf = np.empty(half, dtype=np.complex128)
    ref = _apply_word_peven(
        psi_np,
        word,
        wires,
        n,
        wire0_bit,
        out=out,
        buf=scratch,
        dtype_name="complex128",
        wire0_buf=wire0_buf,
    )

    # GPU formula: phase[k] = c_phase * sign[k] (sign precomputed as float)
    arange_np = np.arange(half, dtype=np.int64)
    perm = arange_np ^ xor_mask
    ppsi = psi_np[perm]
    if parity_tensor is not None:
        sign = parity_tensor.numpy().astype(np.float64)
        ppsi = ppsi * sign
    ppsi = ppsi * c_phase

    np.testing.assert_allclose(
        ppsi.real, ref.real, atol=1e-12, err_msg=f"real mismatch for {word} on {wires}"
    )
    np.testing.assert_allclose(
        ppsi.imag, ref.imag, atol=1e-12, err_msg=f"imag mismatch for {word} on {wires}"
    )


def test_gate_factor_peven_caches_sign_float():
    """sign_float (not just the raw int8 parity tensor) must be cached per sign_mask.

    Regression test for a peven-GPU memory bug:
    the int8 parity tensor was deduplicated by (sign_mask, half, device), but
    sign_float = 1 - 2*parity.float() -- the tensor actually retained in
    _op_data -- was recomputed (freshly allocated) on every call, so the
    per-operator memory blowup the int8 dedup was meant to prevent still
    happened one derivation step later. theta is varied between calls to
    confirm the cache key is sign_mask-only, not call-specific.
    """
    cache: dict = {}
    n = 6
    half = 1 << (n - 1)
    device = torch.device("cpu")

    _, sign_a, _, _, _ = _precompute_gate_factor("ZZ", (1, 2), 0.0, n, half, device, cache=cache)
    _, sign_b, _, _, _ = _precompute_gate_factor("ZZ", (1, 2), 0.7, n, half, device, cache=cache)

    assert sign_a is not None
    assert sign_a is sign_b, "sign_float tensor was reallocated on a cache hit"


@pytest.mark.parametrize(
    "n,word,wires",
    [
        (6, "YZ", (0, 1)),
        (6, "ZY", (1, 0)),
        (6, "XY", (0, 1)),
        (6, "YX", (1, 0)),
        (8, "YYZ", (0, 2, 3)),
        (8, "XYY", (1, 0, 2)),
        (8, "ZYZ", (2, 0, 3)),
        (6, "ZZ", (0, 1)),
        (6, "XZ", (0, 1)),
        (6, "ZX", (1, 0)),
        (6, "ZZ", (0, 2)),
        (6, "YZ", (2, 0)),
        (6, "YY", (1, 2)),
        (6, "YYX", (1, 2, 3)),
    ],
)
def test_gate_factor_sign_comprehensive(n, word, wires):
    """Finding 8: exhaustive sign validation against reference peven evaluator."""
    from _reference_slow_statevec import _apply_word_peven

    from spingqe_lmg.core.parity import build_even_table

    c_to_full, _, wire0_bit = build_even_table(n)
    half = 1 << (n - 1)
    rng = np.random.default_rng(99)
    psi_np = (rng.random(half) + 1j * rng.random(half)).astype(np.complex128)
    psi_np /= np.linalg.norm(psi_np)
    psi_np_orig = psi_np.copy()

    xor_mask, parity_tensor, c_phase, _, _ = _precompute_gate_factor(
        word, wires, 0.0, n, half, torch.device("cpu")
    )

    scratch = np.empty(half, dtype=np.complex128)
    out = np.empty(half, dtype=np.complex128)
    wire0_buf = np.empty(half, dtype=np.complex128)
    ref = _apply_word_peven(
        psi_np_orig,
        word,
        wires,
        n,
        wire0_bit,
        out=out,
        buf=scratch,
        dtype_name="complex128",
        wire0_buf=wire0_buf,
    )

    arange_np = np.arange(half, dtype=np.int64)
    perm = arange_np ^ xor_mask
    ppsi = psi_np[perm].copy()
    if parity_tensor is not None:
        sign = parity_tensor.numpy().astype(np.float64)
        ppsi = ppsi * sign
    ppsi = ppsi * c_phase

    np.testing.assert_allclose(
        ppsi.real,
        ref.real,
        atol=1e-12,
        err_msg=f"real mismatch for {word} on {wires}",
    )
    np.testing.assert_allclose(
        ppsi.imag,
        ref.imag,
        atol=1e-12,
        err_msg=f"imag mismatch for {word} on {wires}",
    )


def test_parity_even_gpu_gate_factor_is_float_sign():
    """Precomputed parity_tensor in _op_data must be a float sign tensor,
    not raw int8. Verifies the allocation is done once at precompute time."""
    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=2.0)
    pool = build_pauli_pair_pool(n)
    ev = ParityEvenGPUEvaluator(pool, ham, n, dtype=torch.complex64, device="cpu")
    for factors in ev._op_data:
        for factor in factors:
            # factor is (xor_mask, parity_tensor, c_phase, cos_h, sin_h)
            # parity_tensor may be None for all-identity factors
            parity_t = factor[1]
            if parity_t is not None:
                assert parity_t.dtype in (torch.float32, torch.float64), (
                    f"parity tensor dtype should be float, got {parity_t.dtype}"
                )
                # Values must be +1 or -1 (sign tensor, not 0/1 parity)
                assert torch.all((parity_t == 1.0) | (parity_t == -1.0)), (
                    "sign tensor must contain only +1/-1"
                )


# ---------------------------------------------------------------------------
# initial_energy correctness
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("n,lam", [(4, 0.8), (6, 1.5), (8, 2.0)])
def test_initial_energy_matches_peven_cpu(n, lam):
    ham = lmg_hamiltonian(n, h=1.0, lam=lam)
    pool = build_pauli_pair_pool(n)
    ref = ParityEvenEvaluator(pool, ham, n).initial_energy()
    got = ParityEvenGPUEvaluator(pool, ham, n, device=CUDA).initial_energy()
    assert got == pytest.approx(ref, abs=ATOL_FP32)


@pytest.mark.gpu
@pytest.mark.parametrize("n,lam", [(4, 0.8), (6, 1.5)])
def test_initial_energy_fp64_matches_peven_exactly(n, lam):
    ham = lmg_hamiltonian(n, h=1.0, lam=lam)
    pool = build_pauli_pair_pool(n)
    ref = ParityEvenEvaluator(pool, ham, n).initial_energy()
    got = ParityEvenGPUEvaluator(pool, ham, n, dtype="complex128", device=CUDA).initial_energy()
    assert got == pytest.approx(ref, abs=ATOL_FP64)


# ---------------------------------------------------------------------------
# __call__ correctness: pauli-pair pool
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("n", [4, 6])
@pytest.mark.parametrize("lam", [0.8, 1.5])
def test_call_matches_peven_pauli_pair(n, lam):
    """ParityEvenGPUEvaluator (FP32) agrees with CUDA evaluator within FP32 tolerance."""
    ham = lmg_hamiltonian(n, h=1.0, lam=lam)
    pool = build_pauli_pair_pool(n)
    rng = np.random.default_rng(n + int(lam * 10))
    batch = rng.integers(0, len(pool), size=(4, 8))

    ref = ParityEvenEvaluator(pool, ham, n)(batch)
    got = ParityEvenGPUEvaluator(pool, ham, n, device=CUDA)(batch)
    np.testing.assert_allclose(got, ref, atol=ATOL_FP32, err_msg=f"mismatch at N={n}, lam={lam}")


@pytest.mark.gpu
@pytest.mark.parametrize("n,lam", [(4, 0.8), (4, 1.5), (6, 0.8), (6, 1.5)])
def test_call_fp64_matches_peven_pauli_pair(n, lam):
    ham = lmg_hamiltonian(n, h=1.0, lam=lam)
    pool = build_pauli_pair_pool(n)
    rng = np.random.default_rng(n * 7 + int(lam * 100))
    batch = rng.integers(0, len(pool), size=(4, 8))

    ref = ParityEvenEvaluator(pool, ham, n)(batch)
    got = ParityEvenGPUEvaluator(pool, ham, n, dtype="complex128", device=CUDA)(batch)
    np.testing.assert_allclose(
        got, ref, atol=ATOL_FP64, err_msg=f"FP64 mismatch at N={n}, lam={lam}"
    )


# ---------------------------------------------------------------------------
# __call__ correctness: collective pool
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("n", [4, 6])
@pytest.mark.parametrize("lam", [0.8, 2.0])
def test_call_matches_peven_collective(n, lam):
    ham = lmg_hamiltonian(n, h=1.0, lam=lam)
    pool = build_collective_pool(n)
    rng = np.random.default_rng(n * 100 + int(lam * 10))
    batch = rng.integers(0, len(pool), size=(3, 6))

    ref = ParityEvenEvaluator(pool, ham, n)(batch)
    got = ParityEvenGPUEvaluator(pool, ham, n, device=CUDA)(batch)
    np.testing.assert_allclose(
        got, ref, atol=ATOL_FP32, err_msg=f"collective mismatch at N={n}, lam={lam}"
    )


# ---------------------------------------------------------------------------
# ghz_x initial state
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_ghz_x_initial_state():
    n, lam = 4, 1.5
    ham = lmg_hamiltonian(n, h=1.0, lam=lam)
    pool = build_pauli_pair_pool(n)
    rng = np.random.default_rng(0)
    batch = rng.integers(0, len(pool), size=(2, 4))

    ref = ParityEvenEvaluator(pool, ham, n, init_angle="ghz_x")(batch)
    got = ParityEvenGPUEvaluator(pool, ham, n, init_angle="ghz_x", device=CUDA)(batch)
    np.testing.assert_allclose(got, ref, atol=ATOL_FP32)


# ---------------------------------------------------------------------------
# Caching: repeated rows return consistent results
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_cache_deduplication():
    n, lam = 4, 1.5
    ham = lmg_hamiltonian(n, h=1.0, lam=lam)
    pool = build_pauli_pair_pool(n)
    rng = np.random.default_rng(7)
    row = rng.integers(0, len(pool), size=(1, 6))
    batch_dup = np.tile(row, (4, 1))  # same sequence repeated 4 times

    ev = ParityEvenGPUEvaluator(pool, ham, n, device=CUDA)
    result = ev(batch_dup)
    # All rows must be identical
    np.testing.assert_allclose(result[0], result[1])
    np.testing.assert_allclose(result[0], result[3])


# ---------------------------------------------------------------------------
# Parity-rejection tests (same cases as test_statevec.py)
# ---------------------------------------------------------------------------


def test_rejects_parity_odd_pool():
    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=1.5)
    pool = build_pauli_pair_pool(n, paulis=("ZZ", "XX", "YY", "YZ", "XY"))
    with pytest.raises(ValueError, match="parity-odd"):
        ParityEvenGPUEvaluator(pool, ham, n, device=CUDA)


def test_rejects_nonzero_float_init_angle():
    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=2.0)
    pool = build_pauli_pair_pool(n)
    theta = lmg_mean_field_angle(1.0, 2.0)
    assert theta > 0.1
    with pytest.raises(ValueError, match="odd-parity components"):
        ParityEvenGPUEvaluator(pool, ham, n, init_angle=theta, device=CUDA)


# ---------------------------------------------------------------------------
# GPU tests — require a CUDA device
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_gpu_matches_cpu_result():
    """CUDA device produces the same energies as the CUDA evaluator."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    n, lam = 6, 1.5
    ham = lmg_hamiltonian(n, h=1.0, lam=lam)
    pool = build_pauli_pair_pool(n)
    rng = np.random.default_rng(99)
    batch = rng.integers(0, len(pool), size=(4, 8))

    cpu_ev = ParityEvenEvaluator(pool, ham, n)
    gpu_ev = ParityEvenGPUEvaluator(pool, ham, n, device="cuda")
    np.testing.assert_allclose(gpu_ev(batch), cpu_ev(batch), atol=ATOL_FP32)


@pytest.mark.gpu
def test_fused_cuda_kernels_available():
    """Regression guard: the fused CUDA extensions must actually compile and
    run on a GPU node, not silently fall back to the slow Python path. See
    the ninja setup requirements and the TORCH_CUDA_ARCH_LIST default in
    _statevec_cuda.py / _peven_cuda.py.

    Compilation is deferred to first real CUDA use (see the lazy-load fix in
    statevec_gpu.py), so this test must actually run both evaluators on a
    CUDA device before checking the availability flags -- checking them
    right after import would trivially pass without compiling anything.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from spingqe_lmg.evaluators import statevec_gpu

    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=1.0)
    pool = build_pauli_pair_pool(n)
    batch = np.array([[0, 1]])

    incr_ev = statevec_gpu.IncrementalGPUEvaluator(pool, ham, n, device="cuda")
    incr_ev(batch)
    peven_ev = statevec_gpu.ParityEvenGPUEvaluator(pool, ham, n, device="cuda")
    peven_ev(batch)

    assert statevec_gpu.CUDA_FUSED_AVAILABLE, "statevec fused CUDA kernel failed to load"
    assert statevec_gpu.PEVEN_CUDA_FUSED_AVAILABLE, "peven fused CUDA kernel failed to load"


# ---------------------------------------------------------------------------
# Regression: sign_float cache must deduplicate across pool operators
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_parity_even_gpu_evaluator_dedups_sign_float_cache():
    """Constructing the evaluator must cache far fewer tensors than pool operators.

    Before the fix, every pool operator with a nonzero sign_mask allocated
    its own sign_float tensor even when it shared a sign_mask with another
    operator -- so cache size scaled with pool size (~8500 ops at N=24), not
    with the number of *distinct* sign_masks. This checks the ratio directly
    at a moderate N where full construction is cheap regardless of the bug,
    so it isolates the dedup behavior from the OOM itself (covered in the
    next test).
    """
    n = 16
    ham = lmg_hamiltonian(n, h=1.0, lam=2.0)
    pool = build_pauli_pair_pool(n)
    ev = ParityEvenGPUEvaluator(pool, ham, n, dtype=torch.complex64, device="cuda")

    assert len(ev._parity_cache) < len(pool), (
        f"cache holds {len(ev._parity_cache)} entries for {len(pool)} pool "
        "operators -- expected far fewer unique sign_masks than operators"
    )


@pytest.mark.gpu
def test_parity_even_gpu_evaluator_constructs_at_n24():
    """Reproduces the exact reported OOM symptom: construction must succeed at N=24.

    Before the fix this OOM'd inside _precompute_gate_factor_peven while
    building _op_data (94.41 GB already allocated, failing to allocate one
    more 32 MiB tensor).
    """
    n = 24
    ham = lmg_hamiltonian(n, h=1.0, lam=2.0)
    pool = build_pauli_pair_pool(n)

    torch.cuda.reset_peak_memory_stats()
    ev = ParityEvenGPUEvaluator(pool, ham, n, dtype=torch.complex64, device="cuda")
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    assert ev is not None
    assert peak_gb < 40.0, (
        f"construction used {peak_gb:.2f} GB -- expected well under the "
        "94+ GB that OOM'd before the fix"
    )
