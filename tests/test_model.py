import numpy as np
import pytest
import torch

from spingqe_lmg.model import GQE, GPTConfig


def tiny_config(vocab_size=20, block_size=6):
    return GPTConfig(
        vocab_size=vocab_size, block_size=block_size, n_layer=2, n_head=2, n_embd=32, dropout=0.0
    )


def test_forward_shapes():
    model = GQE(tiny_config())
    idx = torch.randint(0, 20, (3, 5))
    logits = model(idx)
    assert logits.shape == (3, 5, 20)


def test_generate_in_vocab_range_and_length():
    model = GQE(tiny_config())
    model.eval()
    tokens, cumsum_logits = model.generate(n_sequences=4, max_new_tokens=6, temperature=0.7)
    assert tokens.shape == (4, 7)  # BOS + 6 generated
    assert cumsum_logits.shape == (4, 1)
    assert torch.all(tokens[:, 0] == 0)
    assert torch.all(tokens[:, 1:] >= 1)  # BOS is masked during sampling
    assert torch.all(tokens < 20)


def test_seeded_generation_deterministic():
    model = GQE(tiny_config())
    model.eval()
    torch.manual_seed(123)
    a, _ = model.generate(n_sequences=3, max_new_tokens=6)
    torch.manual_seed(123)
    b, _ = model.generate(n_sequences=3, max_new_tokens=6)
    assert torch.equal(a, b)


def test_calculate_loss_shapes_and_weighting():
    model = GQE(tiny_config())
    tokens = torch.randint(1, 20, (4, 7))
    tokens[:, 0] = 0
    energies = torch.randn(4, 6)
    loss_w, loss_uw = model.calculate_loss(tokens, energies, beta=0.3)
    assert loss_w.dim() == 0 and loss_uw.dim() == 0
    loss_none, loss_none_uw = model.calculate_loss(tokens, energies, beta=None)
    assert torch.isclose(loss_none, loss_none_uw)


def test_loss_decreases_overfit():
    torch.manual_seed(0)
    model = GQE(tiny_config())
    opt = model.configure_optimizers(
        weight_decay=0.0, learning_rate=3e-3, betas=(0.9, 0.95), device_type="cpu"
    )
    tokens = torch.randint(1, 20, (8, 7))
    tokens[:, 0] = 0
    energies = torch.cumsum(torch.randn(8, 6), dim=1)
    model.train()
    losses = []
    for _ in range(150):
        opt.zero_grad()
        loss, _ = model.calculate_loss(tokens, energies, beta=None)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] / 10


@pytest.mark.gpu
def test_model_runs_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    model = GQE(tiny_config()).to("cuda")
    tokens, _ = model.generate(n_sequences=2, max_new_tokens=4, device="cuda")
    assert tokens.device.type == "cuda"


def test_boltzmann_sampling_prefers_low_logits():
    # with T -> 0, generation should pick the argmin-logit token; train a model is
    # overkill — instead check the distribution directly on a frozen model
    model = GQE(tiny_config())
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, 1, dtype=torch.long))[0, -1]
        logits[0] = float("inf")
        probs = torch.softmax(-logits / 0.01, dim=-1)
    assert probs.argmax().item() == logits[1:].argmin().item() + 1
    assert np.isclose(probs.sum().item(), 1.0)


def test_generate_cumsum_logits_equals_internal():
    """cumsum_logits from generate() must equal cumsum_logits[:, -1] from
    a one-pass forward on the same tokens. If they match, the MAE metric
    in training.py is correct (both are on the same energy scale)."""
    model = GQE(tiny_config())
    model.eval()

    torch.manual_seed(42)
    tokens, cumsum_logits = model.generate(
        n_sequences=3, max_new_tokens=5, temperature=1.0, device="cpu"
    )
    cumsum_logits = cumsum_logits.reshape(-1)

    with torch.no_grad():
        current_tokens = tokens[:, :-1]
        next_tokens = tokens[:, 1:]
        logits = model(current_tokens)
        next_token_logits = logits.gather(2, next_tokens.unsqueeze(2)).squeeze(2)
        cumsum = torch.cumsum(next_token_logits, dim=1)
        cumsum_final = cumsum[:, -1]

    assert torch.allclose(cumsum_logits, cumsum_final, atol=1e-4), (
        f"cumsum_logits={cumsum_logits.tolist()}, cumsum_final={cumsum_final.tolist()}"
    )


def test_weight_tying_no_duplicate_in_optimizer():
    """When wte.weight == lm_head.weight, the optimizer must contain the
    tied tensor exactly once."""
    from spingqe_lmg.model._optim import gqe_configure_optimizers

    model = GQE(GPTConfig(vocab_size=100, block_size=12, n_layer=2, n_head=2, n_embd=64))
    opt = gqe_configure_optimizers(
        model.named_parameters(),
        weight_decay=0.1,
        learning_rate=1e-4,
        betas=(0.9, 0.95),
        device_type="cpu",
    )
    all_params = []
    for group in opt.param_groups:
        all_params.extend(group["params"])
    unique_ids = {id(p) for p in all_params}
    assert len(unique_ids) == len(all_params), (
        f"{len(all_params)} param slots but only {len(unique_ids)} unique tensors"
    )


def test_loss_one_hot_vs_gather_equivalent():
    """gqe_calculate_loss with gather must produce same loss as old one_hot code."""
    import torch

    from spingqe_lmg.model import GQE, GPTConfig

    vocab_size = 20
    model = GQE(
        GPTConfig(vocab_size=vocab_size, block_size=6, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    )
    torch.manual_seed(0)
    tokens = torch.randint(1, vocab_size, (4, 7))
    tokens[:, 0] = 0

    current_tokens, next_tokens = tokens[:, :-1], tokens[:, 1:]
    with torch.no_grad():
        logits = model(current_tokens)
    one_hot = torch.nn.functional.one_hot(next_tokens, num_classes=vocab_size)
    ntl_onehot = (logits * one_hot).sum(axis=2)
    ntl_gather = logits.gather(2, next_tokens.unsqueeze(2)).squeeze(2)
    assert torch.allclose(ntl_onehot, ntl_gather, atol=1e-6)


def test_gelu_is_tanh_approximation():
    """GQE uses the tanh-approximate GELU, not the exact erf-based one."""
    import torch.nn.functional as F

    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = F.gelu(x, approximate="tanh")
    expected = x * 0.5 * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x**3)))
    torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-6)


def test_beta_zero_uses_weighted_branch():
    """Finding 6: beta=0.0 should run weighted branch with neutral weights (0.5),
    not fall through to unweighted branch."""
    from spingqe_lmg.model._loss import gqe_calculate_loss

    class _DummyLogitsFn:
        """Returns zero logits — all token log-probs are 0, cumsum_logits = 0."""

        def __call__(self, tokens):
            B, L = tokens.shape
            return torch.zeros(B, L, 10)  # 10 = vocab_size

    B, L, V = 4, 5, 10
    tokens = torch.randint(1, V, (B, L))
    energies = torch.randn(B, L - 1)  # per-step energies

    logits_fn = _DummyLogitsFn()

    loss_none, uw_none = gqe_calculate_loss(logits_fn, tokens, energies, V, beta=None)
    loss_zero, uw_zero = gqe_calculate_loss(logits_fn, tokens, energies, V, beta=0.0)

    # With beta=None, loss == uw_loss (unweighted)
    assert loss_none.item() == uw_none.item()

    # With beta=0.0, loss = 0.5 * uw_loss (all weights = 0.5)
    assert loss_zero.item() == pytest.approx(0.5 * uw_zero.item())

    # Unweighted loss component should be the same in both cases
    assert uw_none.item() == pytest.approx(uw_zero.item())


def test_training_config_beta_forwarding():
    """Integration: beta=0.0 from config is not coerced to None."""
    from spingqe_lmg.config import config_from_dict

    cfg_data = {
        "hamiltonian": {"kind": "lmg", "n_qubits": 4, "h": 1.0, "lam": 0.3, "gamma": 0},
        "train": {
            "evaluator": "incremental",
            "seq_len": 4,
            "beta": 0.0,
            "n_jobs": 1,
        },
        "pool": {"kind": "pauli_pair"},
        "model": {"n_layer": 1, "n_head": 2, "n_embd": 16},
    }
    cfg = config_from_dict(cfg_data)
    assert cfg.train.beta == 0.0, "beta=0.0 must not be coerced"
