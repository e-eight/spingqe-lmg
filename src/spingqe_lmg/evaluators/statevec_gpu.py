"""GPU-accelerated statevector evaluators (PyTorch + fused CUDA kernels).

``IncrementalGPUEvaluator`` — full ``2^N`` Hilbert space, any pool.
``ParityEvenGPUEvaluator`` — compressed ``2^(N-1)`` parity-even sector.

Gate factors are applied via XOR-gather with precomputed parity signs.
Energy evaluation fuses the diagonal and all off-diagonal Hamiltonian
terms into a single-pass reduction.
"""

from __future__ import annotations

import numpy as np

from spingqe_lmg.core.caching import CachedEvaluatorMixin, LRUCache
from spingqe_lmg.core.parity import build_even_table, build_h_diag, validate_peven_preconditions
from spingqe_lmg.core.statevec_ops import (
    hamiltonian_terms,
    initial_statevector,
    split_sv_terms,
)
from spingqe_lmg.core.torch_utils import (
    parity_np,
    real_counterpart,
    to_torch_dtype,
)
from spingqe_lmg.refine import RefineResult, refine_tokens

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from spingqe_lmg.core.numba_cuda_kernels import apply_gate, eval_energy

    CUDA_FUSED_AVAILABLE = True
    PEVEN_CUDA_FUSED_AVAILABLE = True
except ImportError:
    CUDA_FUSED_AVAILABLE = False
    PEVEN_CUDA_FUSED_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared sign-tensor cache (used by both gate-factor precompute variants below)
# ---------------------------------------------------------------------------


def _cached_sign_tensor(
    cache: dict | None, sign_mask: int, size: int, device: torch.device
) -> torch.Tensor | None:
    """Return the +-1 float sign tensor for *sign_mask*, caching on (sign_mask, size, device).

    Returns None when sign_mask == 0 (no phase to apply). The cached tensor
    (not just the raw int8 parity array) must be reused across calls with the
    same sign_mask, or per-operator memory scales with pool size instead of
    the number of distinct sign_masks (see the N=24 OOM fixed in 4c15940 /
    22893df).
    """
    if sign_mask == 0:
        return None
    if cache is not None:
        cache_key = (sign_mask, size, device)
        if cache_key not in cache:
            raw = torch.as_tensor(parity_np(size, sign_mask), device=device)
            cache[cache_key] = 1.0 - 2.0 * raw.float()
        return cache[cache_key]
    raw = torch.as_tensor(parity_np(size, sign_mask), device=device)
    return 1.0 - 2.0 * raw.float()


# ---------------------------------------------------------------------------
# Pauli word precomputation (full-Hilbert --- no wire-0 special case)
# ---------------------------------------------------------------------------


def _precompute_gate_factor(
    word: str,
    wires: tuple[int, ...],
    theta: float,
    n_qubits: int,
    n2: int,
    device: torch.device,
    cache: dict | None = None,
) -> tuple:
    """Precompute (xor_mask, parity_tensor, c_phase, cos_h, sin_h) for one factor.

    For Pauli word P on the full 2^N Hilbert space:
      P|psi>[k] = c_phase * (-1)^{popcount(k & sign_mask)} * psi[k ^ xor_mask]

    where c_phase = i^{num_Y} and sign_mask accumulates Z/Y wire bits.
    No wire-0 special case (unlike peven_gpu).
    """
    xor_mask = 0
    sign_mask = 0
    num_y = 0
    for letter, wire in zip(word, wires, strict=False):
        a = n_qubits - 1 - int(wire)  # wire → axis (MSB-first)
        if letter == "X":
            xor_mask ^= 1 << a
        elif letter == "Z":
            sign_mask ^= 1 << a
        elif letter == "Y":
            xor_mask ^= 1 << a
            sign_mask ^= 1 << a
            num_y += 1
        # I: no effect

    c_phase = 1j**num_y

    parity_tensor = _cached_sign_tensor(cache, sign_mask, n2, device)

    cos_h = float(np.cos(0.5 * theta))
    sin_h = float(np.sin(0.5 * theta))
    return (xor_mask, parity_tensor, c_phase.real, c_phase.imag, cos_h, sin_h)


def _precompute_offdiag(offdiag_terms, n_qubits, device, real_dtype=None):
    """Return (xor_masks, sign_masks, c_re, c_im) tensors for Hamiltonian terms."""
    xor_list, sign_list, cre_list, cim_list = [], [], [], []
    for coeff, letters in offdiag_terms:
        xor_mask = 0
        sign_mask = 0
        num_y = 0
        for wire, letter_byte in letters:
            ltr = chr(letter_byte)
            a = n_qubits - 1 - int(wire)
            if ltr == "X":
                xor_mask ^= 1 << a
            elif ltr == "Z":
                sign_mask ^= 1 << a
            elif ltr == "Y":
                xor_mask ^= 1 << a
                sign_mask ^= 1 << a
                num_y += 1
        c_total = complex(coeff) * (1j**num_y)
        xor_list.append(xor_mask)
        sign_list.append(sign_mask)
        cre_list.append(c_total.real)
        cim_list.append(c_total.imag)

    if real_dtype is None:
        real_dtype = torch.float32 if device.type != "cpu" else torch.float64
    return (
        torch.tensor(xor_list, dtype=torch.int64, device=device),
        torch.tensor(sign_list, dtype=torch.int64, device=device),
        torch.tensor(cre_list, dtype=real_dtype, device=device),
        torch.tensor(cim_list, dtype=real_dtype, device=device),
    )


# ---------------------------------------------------------------------------
# Parity-even gate/energy precomputation (2^(N-1) compressed basis)
# ---------------------------------------------------------------------------


def _precompute_gate_factor_peven(
    word: str,
    wires: tuple,
    theta: float,
    n_qubits: int,
    half: int,
    device: torch.device,
    cache: dict | None = None,
) -> tuple:
    """Precompute (xor_mask, parity_tensor, c_phase, cos_h, sin_h) for one gate factor.

    For parity-even word P, the compressed-basis gate action is:
        (P|ψ⟩)[k] = c_phase · sign[k] · ψ[k ⊕ xor_mask]
    where sign[k] = (−1)^{popcount(k & sign_mask) mod 2} is ±1.

    Phase derivation (matches _apply_word_peven exactly):
    - X at wire w>0  : xor_mask |= 1<<(nc-w);  no phase contribution.
    - Y at wire w>0  : xor_mask |= 1<<(nc-w);  contributes (−1)^{1−bit_{nc-w}(k)}
                       at destination k → sign_mask |= 1<<(nc-w), real_sign ×= −1.
    - Z at wire w>0  : contributes (−1)^{bit_{nc-w}(k)} → sign_mask |= 1<<(nc-w).
    - Z/Y at wire 0  : wire-0 phase uses SOURCE index k⊕xor_mask.
                       (−1)^{wire0_bit[k⊕xmask]} = (−1)^{popcount(k)%2}·(−1)^{popcount(xmask)%2}
                       → sign_mask ^= all_nc_bits; real_sign ×= (−1)^{popcount(xor_mask)%2}.
    Complex constant: c_phase = (1j)^{num_Y} · real_sign   (always ±1 or ±i).
    parity_tensor[k] = popcount(k & sign_mask) % 2  stored as int8 on device.
    """
    nc = n_qubits - 1

    xor_mask = 0
    for letter, wire in zip(word, wires, strict=False):
        if letter in ("X", "Y") and wire > 0:
            xor_mask ^= 1 << (nc - wire)

    sign_mask = 0
    real_sign = 1
    for letter, wire in zip(word, wires, strict=False):
        if wire > 0:
            if letter == "Z":
                sign_mask ^= 1 << (nc - wire)
            elif letter == "Y":
                sign_mask ^= 1 << (nc - wire)
                real_sign *= -1
        else:  # wire == 0, only Z/Y carry a phase
            if letter in ("Z", "Y"):
                sign_mask ^= (1 << nc) - 1  # all nc bits → encodes popcount(k) % 2
                xor_pc_parity = bin(xor_mask).count("1") % 2
                real_sign *= (-1) ** xor_pc_parity

    num_y = sum(1 for ltr in word if ltr == "Y")
    c_phase = (1j**num_y) * real_sign  # Python complex; ±1 for even num_Y

    parity_tensor = _cached_sign_tensor(cache, sign_mask, half, device)

    cos_h = float(np.cos(0.5 * theta))
    sin_h = float(np.sin(0.5 * theta))
    return (xor_mask, parity_tensor, c_phase, cos_h, sin_h)


def _precompute_ham_offdiag_term(
    coeff: float,
    letters: tuple,
    n_qubits: int,
    half: int,
    device: torch.device,
) -> tuple:
    """Compact energy representation for one off-diagonal Hamiltonian term.

    Returns ``(xor_mask, sign_mask, c_total)``.  *sign_mask* is an integer;
    the CUDA kernel computes parity on-the-fly from it via
    ``(k & sign_mask).bit_count() & 1``, avoiding the O(K_off x 2^(N-1))
    precomputed float tensor.
    """
    nc = n_qubits - 1
    word = "".join(ltr for _w, ltr in letters)
    wires = tuple(w for w, _ltr in letters)

    xor_mask = 0
    for ltr, w in zip(word, wires, strict=False):
        if ltr in ("X", "Y") and w > 0:
            xor_mask ^= 1 << (nc - w)

    y_wires = [w for ltr, w in zip(word, wires, strict=False) if ltr == "Y"]
    num_y = len(y_wires)
    sign_mask = 0
    for w in y_wires:
        if w == 0:
            sign_mask ^= (1 << nc) - 1
        else:
            sign_mask ^= 1 << (nc - w)

    c_total = complex(coeff) * (1j**num_y)
    return (xor_mask, sign_mask, c_total)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class IncrementalGPUEvaluator(CachedEvaluatorMixin):
    """Incremental statevector evaluator (GPU, full 2^N Hilbert space, PyTorch).

    GPU-accelerated drop-in for IncrementalEvaluator.  Called the "incremental
    GPU evaluator" in the manuscript.

    Parameters
    ----------
    pool, ham, n_qubits, init_angle
        Same as ``IncrementalEvaluator``.  ``n_jobs`` accepted but ignored.
    dtype : torch / numpy dtype, optional
        Complex64 by default for throughput.
    device : str or torch.device, optional
        Defaults to ``"cuda"`` when available.
    cache_maxsize : int or None
        LRU cache bound.
    """

    def _common_gpu_init(
        self,
        pool: list,
        n_qubits: int,
        init_angle,
        dtype,
        device,
        cache_maxsize: int | None,
    ) -> tuple:
        """Set shared attrs; return (np_cplx, np_real, torch_real_dtype).

        Sets: self.device, self.dtype, self._parity_cache, self.n_qubits,
              self.init_angle, self.pool, self._cache.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch is not installed")
        if n_qubits < 2:
            raise ValueError("require n_qubits >= 2")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        td = torch.complex64 if dtype is None else to_torch_dtype(dtype)
        real_td = real_counterpart(td)
        np_cplx = np.complex64 if td == torch.complex64 else np.complex128
        np_real = np.float32 if td == torch.complex64 else np.float64
        self.device = device
        self.dtype = td
        self._parity_cache: dict = {}
        self.n_qubits = n_qubits
        self.init_angle = init_angle
        self.pool = list(pool)
        self._cache: LRUCache = LRUCache(cache_maxsize)
        return np_cplx, np_real, real_td

    def _precompute_gate_factor(self, word: str, wires: tuple, theta: float) -> tuple:
        """Precompute gate factor for full 2^N Hilbert space."""
        n2 = 1 << self.n_qubits
        return _precompute_gate_factor(  # module-level function
            word, wires, theta, self.n_qubits, n2, self.device, self._parity_cache
        )

    def __init__(
        self,
        pool: list,
        ham,
        n_qubits: int,
        init_angle: float | str = 0.0,
        n_jobs: int = 1,
        dtype=None,
        device=None,
        cache_maxsize: int | None = None,
    ):
        np_cplx, np_real, real_td = self._common_gpu_init(
            pool, n_qubits, init_angle, dtype, device, cache_maxsize
        )
        self.ham = ham

        psi0_np = initial_statevector(n_qubits, init_angle, np.dtype(np_cplx))
        self._psi0 = torch.as_tensor(np.ascontiguousarray(psi0_np), device=self.device)
        self._arange = torch.arange(1 << n_qubits, dtype=torch.int64, device=self.device)

        terms = hamiltonian_terms(ham)
        identity_c, h_diag, offdiag_terms = split_sv_terms(terms, n_qubits)
        self._identity_coeff = float(identity_c)
        self._h_diag = torch.as_tensor(np.array(h_diag, dtype=np_real), device=self.device)
        if offdiag_terms:
            self._offdiag = _precompute_offdiag(
                offdiag_terms, n_qubits, self.device, real_dtype=real_td
            )
            self._K_off = self._offdiag[0].shape[0]
        else:
            self._offdiag = None
            self._K_off = 0

        if self._K_off > 0:
            offdiag_xor, offdiag_sign, offdiag_cre, offdiag_cim = self._offdiag
            self._offdiag_xor_gpu = offdiag_xor
            self._offdiag_sign_gpu = offdiag_sign
            self._offdiag_cre_gpu = offdiag_cre
            self._offdiag_cim_gpu = offdiag_cim
        else:
            self._offdiag_xor_gpu = None
            self._offdiag_sign_gpu = None
            self._offdiag_cre_gpu = None
            self._offdiag_cim_gpu = None

        self._op_data: list[list[tuple]] = [
            [
                self._precompute_gate_factor(word, wires, theta)
                for word, wires, theta in op.factors()
            ]
            for op in pool
        ]

    # ------------------------------------------------------------------
    # Core GPU operations
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _apply_factor(
        self,
        psi: torch.Tensor,
        xor_mask: int,
        parity_tensor: torch.Tensor | None,
        c_phase_re: float,
        c_phase_im: float,
        cos_h: float,
        sin_h: float,
    ) -> torch.Tensor:
        if parity_tensor is not None and parity_tensor.numel() > 0:
            parity_arg = parity_tensor
        else:
            parity_arg = torch.empty(0, dtype=torch.float32, device=psi.device)
        return apply_gate(
            psi,
            xor_mask,
            parity_arg,
            c_phase_re,
            c_phase_im,
            cos_h,
            sin_h,
        )

    @torch.no_grad()
    def _eval_energy(self, psi: torch.Tensor) -> float:
        if self._offdiag_xor_gpu is not None:
            return float(
                eval_energy(
                    psi,
                    self._identity_coeff,
                    self._h_diag,
                    self._offdiag_xor_gpu,
                    self._offdiag_sign_gpu,
                    self._offdiag_cre_gpu,
                    self._offdiag_cim_gpu,
                ).item()
            )
        prob = psi.real.pow(2) + psi.imag.pow(2)
        return self._identity_coeff + torch.dot(self._h_diag, prob).item()

    # ------------------------------------------------------------------
    # Sequence evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sequence_energies(self, key: tuple) -> np.ndarray:
        psi = self._psi0.clone()
        energies = []
        for i in key:
            for factor in self._op_data[i]:
                psi = self._apply_factor(psi, *factor)
            energies.append(self._eval_energy(psi))
        return np.array(energies, dtype=np.float64)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

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

    def initial_energy(self) -> float:
        with torch.no_grad():
            return self._eval_energy(self._psi0)


# ---------------------------------------------------------------------------
# Parity-even evaluator (2^(N-1) compressed basis)
# ---------------------------------------------------------------------------


class ParityEvenGPUEvaluator(IncrementalGPUEvaluator):
    """Parity-even statevector evaluator (GPU, 2^(N-1) compressed basis, PyTorch).

    GPU-accelerated drop-in for ParityEvenEvaluator.  Called the "parity-even
    GPU evaluator" in the manuscript.
    """

    def __init__(
        self,
        pool: list,
        ham,
        n_qubits: int,
        init_angle: float | str = 0.0,
        n_jobs: int = 1,
        dtype=None,
        device=None,
        cache_maxsize: int | None = None,
    ):
        if n_qubits < 2:
            raise ValueError("ParityEvenGPUEvaluator requires n_qubits >= 2")
        np_cplx, np_real, real_td = self._common_gpu_init(
            pool, n_qubits, init_angle, dtype, device, cache_maxsize
        )
        self.ham = ham
        half = 1 << (n_qubits - 1)

        # Parity-even tables
        c_to_full, full_to_c, wire0_bit = build_even_table(n_qubits)
        self._c_to_full = c_to_full
        self._wire0_bit = wire0_bit

        # Validate preconditions
        terms = hamiltonian_terms(ham)
        validate_peven_preconditions(
            pool, terms, n_qubits, init_angle, c_to_full, "ParityEvenGPUEvaluator"
        )

        # Compressed initial state on device
        psi_full = initial_statevector(n_qubits, init_angle, np.dtype(np_cplx))
        psi0_np = np.ascontiguousarray(psi_full[c_to_full], dtype=np_cplx)
        self._psi0 = torch.as_tensor(psi0_np, device=self.device)

        # Gather index (avoids per-call .long() cast)
        self._arange = torch.arange(half, dtype=torch.int64, device=self.device)

        # Hamiltonian precomputations (parity-even basis)
        identity_coeff, h_diag, offdiag_terms = build_h_diag(terms, n_qubits, wire0_bit)
        self._identity_coeff = float(identity_coeff)
        self._h_diag_gpu = torch.as_tensor(np.array(h_diag, dtype=np_real), device=self.device)

        if offdiag_terms:
            half = 1 << (n_qubits - 1)
            xors = []
            cre_list = []
            cim_list = []
            sign_list = []
            for coeff, letters in offdiag_terms:
                xor_mask, sign_mask, c_total = _precompute_ham_offdiag_term(
                    coeff, letters, n_qubits, half, self.device
                )
                xors.append(xor_mask)
                sign_list.append(sign_mask)
                cre_list.append(c_total.real)
                cim_list.append(c_total.imag)
            self._K_off = len(offdiag_terms)
            self._offdiag_xor_gpu = torch.tensor(xors, dtype=torch.int64, device=self.device)
            self._offdiag_sign_gpu = torch.tensor(sign_list, dtype=torch.int64, device=self.device)
            self._offdiag_cre_gpu = torch.tensor(cre_list, dtype=real_td, device=self.device)
            self._offdiag_cim_gpu = torch.tensor(cim_list, dtype=real_td, device=self.device)
        else:
            self._K_off = 0
            self._offdiag_xor_gpu = None
            self._offdiag_sign_gpu = None
            self._offdiag_cre_gpu = None
            self._offdiag_cim_gpu = None

        # Pool gate data using peven gate factors
        self._op_data: list[list[tuple]] = [
            [
                self._precompute_gate_factor(word, wires, theta)
                for word, wires, theta in op.factors()
            ]
            for op in pool
        ]

    def _precompute_gate_factor(self, word: str, wires: tuple, theta: float) -> tuple:
        half = 1 << (self.n_qubits - 1)
        return _precompute_gate_factor_peven(
            word, wires, theta, self.n_qubits, half, self.device, self._parity_cache
        )

    @torch.no_grad()
    def _apply_gate(
        self,
        psi: torch.Tensor,
        xor_mask: int,
        parity_tensor: torch.Tensor | None,
        c_phase: complex,
        cos_h: float,
        sin_h: float,
    ) -> torch.Tensor:
        if parity_tensor is not None and parity_tensor.numel() > 0:
            parity_arg = parity_tensor
        else:
            parity_arg = torch.empty(0, dtype=torch.float32, device=psi.device)
        return apply_gate(
            psi,
            xor_mask,
            parity_arg,
            c_phase.real,
            c_phase.imag,
            cos_h,
            sin_h,
        )

    @torch.no_grad()
    def _eval_energy(self, psi: torch.Tensor) -> float:
        if self._offdiag_xor_gpu is not None:
            return float(
                eval_energy(
                    psi,
                    self._identity_coeff,
                    self._h_diag_gpu,
                    self._offdiag_xor_gpu,
                    self._offdiag_sign_gpu,
                    self._offdiag_cre_gpu,
                    self._offdiag_cim_gpu,
                ).item()
            )
        prob = psi.real.pow(2) + psi.imag.pow(2)
        return self._identity_coeff + torch.dot(self._h_diag_gpu, prob).item()

    @torch.no_grad()
    def _sequence_energies(self, key: tuple) -> np.ndarray:
        psi = self._psi0.clone()
        energies = []
        for i in key:
            for factor in self._op_data[i]:
                psi = self._apply_gate(psi, *factor)
            energies.append(self._eval_energy(psi))
        return np.array(energies, dtype=np.float64)

    def initial_energy(self) -> float:
        with torch.no_grad():
            return self._eval_energy(self._psi0)
