"""Energy evaluator backends.

Each evaluator implements the same contract:
    ``evaluator(idx_batch: np.ndarray) -> np.ndarray``
        Takes pool-index batches of shape (B, L) and returns per-step energies
        of shape (B, L).
    ``evaluator.initial_energy() -> float``
        Energy of the reference (empty-circuit) state.
"""

__all__ = [
    "CudaQEvaluator",
    "DickeEvaluator",
    "DickeSpace",
    "PennyLaneEvaluator",
    "PennyLaneGroupedEvaluator",
    "EvaluatorProtocol",
    "IncrementalEvaluator",
    "ParityEvenEvaluator",
    "IncrementalGPUEvaluator",
    "ParityEvenGPUEvaluator",
    "build_evaluator",
    "dicke_refine_tokens",
    "dicke_state_metrics",
]

from typing import Protocol, runtime_checkable

import numpy as np

from spingqe_lmg.config import PennyLaneConfig
from spingqe_lmg.enums import EvaluatorKind, HamiltonianKind
from spingqe_lmg.evaluators.cudaq import CudaQEvaluator
from spingqe_lmg.evaluators.dicke import (
    DickeEvaluator,
    DickeSpace,
    dicke_refine_tokens,
    dicke_state_metrics,
)
from spingqe_lmg.evaluators.pennylane import PennyLaneEvaluator, PennyLaneGroupedEvaluator
from spingqe_lmg.evaluators.statevec import IncrementalEvaluator, ParityEvenEvaluator
from spingqe_lmg.evaluators.statevec_gpu import (
    IncrementalGPUEvaluator,
    ParityEvenGPUEvaluator,
)
from spingqe_lmg.exact import ground_energy
from spingqe_lmg.refine import RefineResult


@runtime_checkable
class EvaluatorProtocol(Protocol):
    """Interface for energy evaluators in the GQE training loop."""

    def __call__(self, idx_batch: np.ndarray) -> np.ndarray:
        """Per-step energies for each sequence in the batch, shape (B, L)."""
        ...

    def initial_energy(self) -> float:
        """Energy of the reference state (no gates applied)."""
        ...

    def refine_tokens(
        self,
        tokens: list[int],
        *,
        qml_device: str = "default.qubit",
        shots: int = 0,
        seed: int = 0,
        optimizer=None,
        gradient=None,
    ) -> RefineResult: ...


_EVALUATOR_REGISTRY: dict[EvaluatorKind, type] = {
    EvaluatorKind.INCREMENTAL: IncrementalEvaluator,
    EvaluatorKind.PARITY_EVEN: ParityEvenEvaluator,
    EvaluatorKind.INCREMENTAL_GPU: IncrementalGPUEvaluator,
    EvaluatorKind.PARITY_EVEN_GPU: ParityEvenGPUEvaluator,
    EvaluatorKind.DICKE: DickeEvaluator,
    EvaluatorKind.CUDAQ: CudaQEvaluator,
    EvaluatorKind.PENNYLANE: PennyLaneEvaluator,
}


def build_evaluator(cfg, pool, ham, init_angle, n_qubits):
    """Construct the evaluator and return (evaluator, ground_energy)."""
    tc = cfg.train
    hc = cfg.hamiltonian
    kind = tc.evaluator

    if kind not in _EVALUATOR_REGISTRY:
        raise ValueError(f"Unknown evaluator kind: {kind!r}")

    cls = _EVALUATOR_REGISTRY[kind]

    kwargs: dict = {}
    if kind in (EvaluatorKind.INCREMENTAL, EvaluatorKind.PARITY_EVEN):
        kwargs = {"n_jobs": tc.n_jobs}
    elif kind is EvaluatorKind.DICKE:
        if hc.kind is not HamiltonianKind.LMG:
            raise ValueError("evaluator='dicke' requires the LMG Hamiltonian")
        kwargs = {"n_jobs": tc.n_jobs}
    elif kind is EvaluatorKind.CUDAQ:
        kwargs = {"n_jobs": tc.n_jobs}
    elif kind is EvaluatorKind.PENNYLANE:
        pl = tc.pennylane or PennyLaneConfig()
        kwargs = {
            "device_name": pl.device,
            "n_jobs": tc.n_jobs,
            "shots": pl.shots,
            "seed": tc.seed,
        }
    elif kind is EvaluatorKind.INCREMENTAL_GPU:
        kwargs = {}
    elif kind is EvaluatorKind.PARITY_EVEN_GPU:
        kwargs = {}
    else:
        raise ValueError(
            f"build_evaluator: unhandled EvaluatorKind {kind!r}. "
            "Add a branch here when introducing a new evaluator."
        )

    # Dicke evaluator takes h/lam/gamma instead of ham
    if kind is EvaluatorKind.DICKE:
        evaluator = cls(pool, n_qubits, hc.h, hc.lam, hc.gamma, init_angle=init_angle, **kwargs)
    else:
        evaluator = cls(pool, ham, n_qubits, init_angle=init_angle, **kwargs)

    e0 = (
        evaluator.ground_energy  # DickeEvaluator: already eigendecomposed
        if hasattr(evaluator, "ground_energy")
        else ground_energy(hc)  # all other evaluators: compute from config
    )
    return evaluator, e0
