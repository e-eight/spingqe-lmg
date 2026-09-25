"""Correctness and import-safety tests for the numba.cuda evaluator kernels.

Generalizes an earlier single-block, N=8 feasibility check to arbitrary
N. The peven-specific wire-0 phase convention is covered separately in
tests/test_peven_gpu.py (production-fixture comparison), not here.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")


def test_importing_numba_cuda_evaluators_does_not_require_a_gpu():
    """@cuda.jit decoration must not touch the CUDA driver at import time.

    Regression test for the claim in the design doc that numba's Dispatcher
    is import-safe (unlike torch.utils.cpp_extension.load, which needed the
    explicit _load_ops() lazy-singleton guard in _cuda_evaluators.py). Runs
    with CUDA_VISIBLE_DEVICES="" to simulate a GPU-less login node.
    """
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, "-c", "import spingqe_lmg.core.numba_cuda_kernels"],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert result.returncode == 0, (
        f"import failed with no GPU visible (CUDA_VISIBLE_DEVICES=''):\n{result.stderr}"
    )


def _reference_apply_gate(psi_in, xor_mask, sign, c_phase, cos_h, sin_h):
    """Plain-NumPy reference for the same math as _apply_gate_kernel."""
    n = psi_in.shape[0]
    idx = np.arange(n) ^ xor_mask
    ppsi = psi_in[idx]
    if sign is not None:
        ppsi = ppsi * sign
    ppsi = ppsi * c_phase
    return cos_h * psi_in - 1j * sin_h * ppsi


def _reference_eval_energy(psi, h_diag, xor_masks, sign_masks, c_re, c_im, identity_coeff):
    """Plain-NumPy reference for the same math as _eval_energy_kernel."""
    n = psi.shape[0]
    total = float(np.sum(h_diag * np.abs(psi) ** 2))
    for j in range(len(xor_masks)):
        sm = sign_masks[j]
        idx = np.arange(n) ^ xor_masks[j]
        if sm == 0:
            signs = np.ones(n)
        else:
            parity = np.bitwise_count(np.arange(n, dtype=np.uint64) & np.uint64(sm)) & 1
            signs = 1.0 - 2.0 * parity
        psi_src = psi[idx] * signs
        dot = np.conj(psi) * psi_src
        total += float(np.sum(dot.real * c_re[j] - dot.imag * c_im[j]))
    return identity_coeff + total


def _build_kernel_test_case(n_qubits, seed=42):
    """Synthetic statevector + 3 offdiag Hamiltonian terms, generalized from
    the feasibility spike's N=8 case to arbitrary (multi-block) N."""
    rng = np.random.default_rng(seed)
    n = 1 << n_qubits
    psi = rng.normal(size=n) + 1j * rng.normal(size=n)
    psi = (psi / np.linalg.norm(psi)).astype(np.complex64)
    h_diag = rng.normal(size=n).astype(np.float32)
    identity_coeff = 0.37
    k_off = 3
    xor_masks = rng.integers(1, n, size=k_off, dtype=np.int64)
    sign_masks = rng.integers(1, n, size=k_off, dtype=np.int64)
    c_re = rng.normal(size=k_off).astype(np.float32)
    c_im = rng.normal(size=k_off).astype(np.float32)
    gate_xor_mask = int(rng.integers(1, n))
    gate_sign = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
    gate_c_phase = complex(0.0, 1.0)
    theta = 0.83
    return {
        "psi": psi,
        "h_diag": h_diag,
        "identity_coeff": identity_coeff,
        "xor_masks": xor_masks,
        "sign_masks": sign_masks,
        "c_re": c_re,
        "c_im": c_im,
        "gate_xor_mask": gate_xor_mask,
        "gate_sign": gate_sign,
        "gate_c_phase": gate_c_phase,
        "cos_h": float(np.cos(0.5 * theta)),
        "sin_h": float(np.sin(0.5 * theta)),
    }


@pytest.mark.gpu
@pytest.mark.parametrize("n_qubits", [8, 10, 12, 14, 16])
def test_apply_gate_matches_numpy_reference(n_qubits):
    from spingqe_lmg.core.numba_cuda_kernels import apply_gate

    tc = _build_kernel_test_case(n_qubits)
    psi_in = torch.as_tensor(tc["psi"], device="cuda")
    sign = torch.as_tensor(tc["gate_sign"], device="cuda")

    got = apply_gate(
        psi_in,
        tc["gate_xor_mask"],
        sign,
        tc["gate_c_phase"].real,
        tc["gate_c_phase"].imag,
        tc["cos_h"],
        tc["sin_h"],
    )
    ref = _reference_apply_gate(
        tc["psi"].astype(np.complex128),
        tc["gate_xor_mask"],
        tc["gate_sign"].astype(np.float64),
        tc["gate_c_phase"],
        tc["cos_h"],
        tc["sin_h"],
    )
    np.testing.assert_allclose(got.cpu().numpy(), ref, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("n_qubits", [8, 10, 12, 14, 16])
def test_eval_energy_matches_numpy_reference(n_qubits):
    from spingqe_lmg.core.numba_cuda_kernels import eval_energy

    tc = _build_kernel_test_case(n_qubits)
    psi = torch.as_tensor(tc["psi"], device="cuda")
    h_diag = torch.as_tensor(tc["h_diag"], device="cuda")
    xor_masks = torch.as_tensor(tc["xor_masks"], device="cuda")
    sign_masks = torch.as_tensor(tc["sign_masks"], device="cuda")
    c_re = torch.as_tensor(tc["c_re"], device="cuda")
    c_im = torch.as_tensor(tc["c_im"], device="cuda")

    got = eval_energy(psi, tc["identity_coeff"], h_diag, xor_masks, sign_masks, c_re, c_im)
    ref = _reference_eval_energy(
        tc["psi"].astype(np.complex128),
        tc["h_diag"].astype(np.float64),
        tc["xor_masks"],
        tc["sign_masks"],
        tc["c_re"].astype(np.float64),
        tc["c_im"].astype(np.float64),
        tc["identity_coeff"],
    )
    assert float(got.item()) == pytest.approx(ref, abs=1e-4)
