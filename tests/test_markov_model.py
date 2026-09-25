import numpy as np
import pytest
import torch

from spingqe_lmg.model import MarkovConfig, MarkovGQE


def tiny_config(vocab_size=20, markov_order=1):
    return MarkovConfig(vocab_size=vocab_size, markov_order=markov_order)


@pytest.mark.parametrize("order", [1, 2])
def test_forward_shapes(order):
    model = MarkovGQE(tiny_config(markov_order=order))
    idx = torch.randint(0, 20, (3, 6))
    logits = model(idx)
    assert logits.shape == (3, 6, 20)


@pytest.mark.parametrize("order", [1, 2])
def test_generate_in_vocab_range_and_length(order):
    model = MarkovGQE(tiny_config(markov_order=order))
    model.eval()
    tokens, cumsum_logits = model.generate(n_sequences=4, max_new_tokens=6, temperature=0.7)
    assert tokens.shape == (4, 7)
    assert cumsum_logits.shape == (4, 1)
    assert torch.all(tokens[:, 0] == 0)
    assert torch.all(tokens[:, 1:] >= 1)
    assert torch.all(tokens < 20)


@pytest.mark.parametrize("order", [1, 2])
def test_seeded_generation_deterministic(order):
    model = MarkovGQE(tiny_config(markov_order=order))
    model.eval()
    torch.manual_seed(123)
    a, _ = model.generate(n_sequences=3, max_new_tokens=6)
    torch.manual_seed(123)
    b, _ = model.generate(n_sequences=3, max_new_tokens=6)
    assert torch.equal(a, b)


@pytest.mark.parametrize("order", [1, 2])
def test_calculate_loss_shapes_and_weighting(order):
    model = MarkovGQE(tiny_config(markov_order=order))
    tokens = torch.randint(1, 20, (4, 7))
    tokens[:, 0] = 0
    energies = torch.randn(4, 6)
    loss_w, loss_uw = model.calculate_loss(tokens, energies, beta=0.3)
    assert loss_w.dim() == 0 and loss_uw.dim() == 0
    loss_none, loss_none_uw = model.calculate_loss(tokens, energies, beta=None)
    assert torch.isclose(loss_none, loss_none_uw)


@pytest.mark.slow
@pytest.mark.parametrize("order", [1, 2])
def test_loss_decreases_overfit(order):
    """Markov models are position-blind — use batched identical sequences so
    position ambiguity does not create conflicting targets."""
    torch.manual_seed(0)
    model = MarkovGQE(tiny_config(markov_order=order))
    opt = model.configure_optimizers(
        weight_decay=0.0, learning_rate=1e-2, betas=(0.9, 0.95), device_type="cpu"
    )
    # single sequence repeated 8× — every position has a unique token, so
    # the model can assign the correct per-token logits without confusion
    base = torch.randint(1, 20, (1, 7))
    base[:, 0] = 0
    tokens = base.repeat(8, 1)
    energies = torch.cumsum(torch.randn(1, 6), dim=1).repeat(8, 1)
    model.train()
    losses = []
    for _ in range(400):
        opt.zero_grad()
        loss, _ = model.calculate_loss(tokens, energies, beta=None)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] / 10


@pytest.mark.parametrize("order", [1, 2])
def test_boltzmann_sampling_prefers_low_logits(order):
    model = MarkovGQE(tiny_config(markov_order=order))
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, 1, dtype=torch.long))[0, -1]
        logits[0] = float("inf")
        probs = torch.softmax(-logits / 0.01, dim=-1)
    assert probs.argmax().item() == logits[1:].argmin().item() + 1
    assert np.isclose(probs.sum().item(), 1.0)


def test_markov1_position_blind():
    """Shifting the input by one position gives shifted-identical logits."""
    model = MarkovGQE(tiny_config(markov_order=1))
    model.eval()
    seq_a = torch.randint(0, 20, (2, 5))
    seq_b = torch.roll(seq_a, shifts=1, dims=1)
    seq_b[:, 0] = seq_a[:, -1]
    logits_a = model(seq_a)
    logits_b = model(seq_b)
    assert torch.allclose(logits_b[:, 1:, :], logits_a[:, :-1, :])


def test_markov2_two_tokens_matter():
    """Same current token but different previous token → different next-token logits."""
    model = MarkovGQE(tiny_config(markov_order=2))
    model.eval()
    ctx_a = torch.tensor([[0, 5]], dtype=torch.long)
    ctx_b = torch.tensor([[3, 5]], dtype=torch.long)
    la = model(ctx_a)[0, -1, :]
    lb = model(ctx_b)[0, -1, :]
    assert not torch.allclose(la, lb)


@pytest.mark.parametrize("order", [1, 2])
def test_factory_builds_correct_model(order):
    from spingqe_lmg.enums import ModelKind
    from spingqe_lmg.model.factory import build_model

    cfg = type(
        "cfg",
        (),
        {
            "kind": ModelKind.MARKOV,
            "order": order,
            "n_layer": 0,
            "n_head": 0,
            "n_embd": 0,
            "dropout": 0.0,
            "bias": False,
        },
    )()
    model = build_model(cfg, vocab_size=20, block_size=12)
    assert isinstance(model, MarkovGQE)
    assert model.config.markov_order == order


@pytest.mark.parametrize("order", [1, 2])
def test_next_logits_matches_forward(order):
    """next_logits(seq) should equal forward(seq)[:, -1, :]."""
    model = MarkovGQE(tiny_config(markov_order=order))
    model.eval()
    seq = torch.randint(0, 20, (3, 4))
    from_forward = model(seq)[:, -1, :]
    from_next = model.model.next_logits(seq)
    assert torch.allclose(from_forward, from_next)
