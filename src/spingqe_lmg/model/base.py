"""GQEModel protocol: the contract between models and the GQE training loop.

Any ``torch.nn.Module`` implementing these three methods can be used with
``spingqe-train``. Standard ``nn.Module`` methods (``train()``, ``eval()``,
``to()``, ``state_dict()``, ``load_state_dict()``) come from the base class.
"""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class GQEModel(Protocol):
    """Interface for autoregressive sequence models in the GQE training loop.

    Implementations must be ``torch.nn.Module`` subclasses.
    """

    @torch.no_grad()
    def generate(
        self, n_sequences: int, max_new_tokens: int, temperature: float, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample sequences autoregressively.

        Returns ``(tokens, summed_logits)`` where ``tokens`` has shape
        ``(n_sequences, max_new_tokens + 1)`` including the BOS column at
        position 0, and ``summed_logits`` are the per-sequence accumulated
        log-probability surrogates (shape ``(n_sequences, 1)``).
        """
        ...

    def calculate_loss(
        self, tokens: torch.Tensor, energies: torch.Tensor, beta: float | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ``(weighted loss, unweighted loss)``.

        ``tokens``: shape ``(B, L+1)`` including the BOS column.
        ``energies``: per-step circuit energies, shape ``(B, L)``.
        ``beta``: sigmoid energy-weighting strength; ``None`` disables weighting.
        """
        ...

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        """Build the optimizer (AdamW) with model-specific parameter groups."""
        ...
