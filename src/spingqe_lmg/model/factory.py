"""Model factory: creates the right architecture based on config.model.kind."""

from spingqe_lmg.enums import ModelKind
from spingqe_lmg.model.base import GQEModel


def build_model(model_cfg, vocab_size: int, block_size: int) -> GQEModel:
    """Create a GQE model from configuration."""
    kind = model_cfg.kind

    if kind is ModelKind.GPT:
        from spingqe_lmg.model.gpt import GPTConfig
        from spingqe_lmg.model.gqe import GQE

        return GQE(
            GPTConfig(
                vocab_size=vocab_size,
                block_size=block_size,
                n_layer=model_cfg.n_layer,
                n_head=model_cfg.n_head,
                n_embd=model_cfg.n_embd,
                dropout=model_cfg.dropout,
                bias=model_cfg.bias,
            )
        )

    elif kind is ModelKind.MARKOV:
        from spingqe_lmg.model.markov import MarkovConfig, MarkovGQE

        return MarkovGQE(
            MarkovConfig(
                vocab_size=vocab_size,
                markov_order=model_cfg.order,
            )
        )

    raise ValueError(f"unknown model kind: {kind!r}")
