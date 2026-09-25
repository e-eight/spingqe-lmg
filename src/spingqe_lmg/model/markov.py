"""Markov-1 and Markov-2 sequence models for GQE training.

Both models predict the next token from only the most recent k tokens
(k = 1 or 2), using a learned transition-logit tensor.  They implement the
GQEModel protocol so they are drop-in replacements for the transformer in
the training loop.

Markov-1 : ``logits[t] = W[token_{t-1}]``  (V x V matrix, 961 params at V=31)
Markov-2 : ``logits[t] = W[token_{t-2}, token_{t-1}]``  (V x V x V tensor)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from spingqe_lmg.model._generate import gqe_generate
from spingqe_lmg.model._loss import gqe_calculate_loss
from spingqe_lmg.model._optim import gqe_configure_optimizers


@dataclass
class MarkovConfig:
    vocab_size: int
    markov_order: int = 1


class MarkovModel(nn.Module):
    """Transition-logit tensor: next-token logits from the last *order* tokens.

    ``forward(idx)`` returns ``(B, L, V)`` where ``logits[:, t, :]`` predicts
    ``next_tokens[:, t]``, matching the contract expected by ``calculate_loss``.
    ``next_logits(seq)`` returns ``(B, V)`` for one-step generation.
    """

    def __init__(self, config: MarkovConfig):
        super().__init__()
        V = config.vocab_size
        shape = (V,) * (config.markov_order + 1)
        self.W = nn.Parameter(torch.randn(*shape) * 0.02)
        self.config = config

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, L = idx.shape
        if self.config.markov_order == 1:
            return self.W[idx]
        # order == 2
        prev_prev = F.pad(idx[:, :-1], (1, 0), value=0)
        return self.W[prev_prev, idx]

    def next_logits(self, seq: torch.Tensor) -> torch.Tensor:
        """Logits for one more token given the generated sequence so far."""
        if self.config.markov_order == 1:
            return self.W[seq[:, -1]]
        B = seq.size(0)
        if seq.size(1) == 1:  # only BOS — pad with another BOS for the first context
            return self.W[
                torch.zeros(B, dtype=torch.long, device=seq.device),
                seq[:, 0],
            ]
        return self.W[seq[:, -2], seq[:, -1]]


class MarkovGQE(nn.Module):
    """Markov-model wrapper satisfying the GQEModel protocol.

    Delegate ``forward`` and ``next_logits`` to the inner ``MarkovModel``;
    ``calculate_loss`` and ``configure_optimizers`` follow the GQE / nanoGPT
    conventions so the training loop works unmodified.
    """

    def __init__(self, config: MarkovConfig):
        super().__init__()
        self.config = config
        self.model = MarkovModel(config)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.model(idx)

    def calculate_loss(self, tokens, energies, beta=None):
        return gqe_calculate_loss(self, tokens, energies, self.config.vocab_size, beta)

    @torch.no_grad()
    def generate(self, n_sequences, max_new_tokens, temperature=1.0, device="cpu"):
        return gqe_generate(
            self.model.next_logits, n_sequences, max_new_tokens, temperature, device
        )

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        return gqe_configure_optimizers(
            self.named_parameters(),
            weight_decay,
            learning_rate,
            betas,
            device_type,
        )
