"""Shared PyTorch dtype utilities used by GPU evaluator backends."""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def to_torch_dtype(dtype) -> "torch.dtype":
    """Coerce a NumPy or torch dtype to a torch complex dtype."""
    if TORCH_AVAILABLE and isinstance(dtype, torch.dtype):
        return dtype
    d = np.dtype(dtype)
    if d == np.complex64:
        return torch.complex64
    if d == np.complex128:
        return torch.complex128
    raise ValueError(f"Unsupported dtype {dtype!r}; expected complex64 or complex128")


def real_counterpart(torch_dtype: "torch.dtype") -> "torch.dtype":
    """Return the real-valued counterpart of a torch complex dtype."""
    return torch.float32 if torch_dtype == torch.complex64 else torch.float64


def _numpy_real(torch_real_dtype: "torch.dtype") -> np.dtype:
    """NumPy dtype corresponding to a torch real dtype."""
    return np.float32 if torch_real_dtype == torch.float32 else np.float64


def parity_np(size: int, sign_mask: int) -> np.ndarray:
    """int8 array p where p[k] = popcount(k & sign_mask) % 2, k in [0, size)."""
    ks = np.arange(size, dtype=np.int64)
    return np.bitwise_count(ks & np.int64(sign_mask)).astype(np.int8) & np.int8(1)
