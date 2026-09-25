"""GQE: GPT trained to generate low-energy gate sequences.

Adapted from ``SpinGQE.py`` in Mindbeam-AI/SpinGQE (MIT). The loss matches the
cumulative sum of next-token logits against the per-step circuit energies; the
optional sigmoid weighting (beta) emphasizes low-energy sequences.
Generation samples each token from a Boltzmann distribution over logits,
softmax(-logits / temperature), with the BOS token (0) masked out.
"""

import torch

from spingqe_lmg.model._generate import gqe_generate
from spingqe_lmg.model._loss import gqe_calculate_loss
from spingqe_lmg.model.gpt import GPT


class GQE(GPT):
    def forward(self, idx):
        device = idx.device
        _, seq_len = idx.size()
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)

        token_emb = self.transformer.wte(idx)
        position_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(token_emb + position_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

    def calculate_loss(self, tokens, energies, beta=None):
        return gqe_calculate_loss(self, tokens, energies, self.config.vocab_size, beta)

    @torch.no_grad()
    def generate(self, n_sequences, max_new_tokens, temperature=1.0, device="cpu"):
        block_size = self.config.block_size

        def _next_logits(idx):
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            return self(idx_cond)[:, -1, :]

        return gqe_generate(_next_logits, n_sequences, max_new_tokens, temperature, device)
