"""Shared optimizer configuration (nanoGPT weight-decay convention)."""

import inspect

import torch


def gqe_configure_optimizers(
    named_parameters,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
    device_type: str,
) -> torch.optim.Optimizer:
    """Build AdamW with separate weight-decay groups (2D params only).

    Matches the nanoGPT convention: parameters with ``dim() >= 2`` (weights,
    embeddings) get *weight_decay*; biases and layer-norms (``dim() < 2``) get 0.
    """
    param_dict = {pn: p for pn, p in named_parameters if p.requires_grad}
    unique_params = list({id(p): p for p in param_dict.values()}.values())
    decay_params = [p for p in unique_params if p.dim() >= 2]
    nodecay_params = [p for p in unique_params if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
