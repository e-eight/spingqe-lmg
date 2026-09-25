"""Shared GQE loss computation (used by GPT and Markov models)."""

import torch


def gqe_calculate_loss(
    logits_fn,
    tokens: torch.Tensor,
    energies: torch.Tensor,
    vocab_size: int,
    beta: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(weighted loss, unweighted loss) for token sequences and per-step energies.

    *logits_fn* is a callable that takes (B, L) tokens and returns (B, L, V) logits
    (typically ``self`` or ``self.model`` depending on the wrapper).
    """
    current_tokens, next_tokens = tokens[:, :-1], tokens[:, 1:]
    logits = logits_fn(current_tokens)
    next_token_logits = logits.gather(2, next_tokens.unsqueeze(2)).squeeze(2)
    cumsum_logits = torch.cumsum(next_token_logits, dim=1)

    uw_loss = torch.mean(torch.square(cumsum_logits - energies))
    if beta is None:
        return uw_loss, uw_loss
    weights = 1 / (1 + torch.exp(beta * energies))  # no gradient: energies are data
    loss = torch.mean(weights * torch.square(cumsum_logits - energies))
    return loss, uw_loss
