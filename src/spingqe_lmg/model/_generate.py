"""Shared autoregressive sampling (used by GPT and Markov models)."""

import torch
from torch.nn import functional as F


@torch.no_grad()
def gqe_generate(
    next_logits_fn,
    n_sequences: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample sequences from softmax(-logits / T); returns (tokens, cumsum_logits).

    *next_logits_fn* is a callable that takes the current token tensor of shape
    (B, T) and returns logits of shape (B, V) for the next token.  BOS (token 0)
    is masked so it can never be sampled.
    """
    idx = torch.zeros(size=(n_sequences, 1), dtype=torch.long, device=device)
    cumsum_logits = torch.zeros(size=(n_sequences, 1), device=device)
    for _ in range(max_new_tokens):
        logits = next_logits_fn(idx)
        logits[:, 0] = float("inf")  # BOS prob = softmax(-inf) = 0
        probs = F.softmax(-logits / temperature, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        cumsum_logits += torch.gather(logits, index=idx_next, dim=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx, cumsum_logits
