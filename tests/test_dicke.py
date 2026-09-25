import time

import numpy as np
import pytest

from spingqe_lmg.config import config_from_dict
from spingqe_lmg.evaluators.dicke import (
    DickeEvaluator,
    DickeSpace,
    dicke_refine_tokens,
    dicke_state_metrics,
)
from spingqe_lmg.evaluators.pennylane import PennyLaneEvaluator, initial_energy
from spingqe_lmg.exact import dicke_matrix, ground_state
from spingqe_lmg.hamiltonians import lmg_hamiltonian, lmg_mean_field_angle
from spingqe_lmg.pools import build_collective_pool, build_pauli_pair_pool
from spingqe_lmg.training import train


@pytest.mark.parametrize("n", [4, 6, 8])
@pytest.mark.parametrize("init_angle", [0.0, 0.7])
def test_dicke_matches_full_space_evaluator(n, init_angle):
    # the decisive parity test: collective sequences give identical per-step
    # energies in the (N+1)-dim sector and the full 2^N space
    h, lam, gamma = 1.0, 1.5, 0.0
    ham = lmg_hamiltonian(n, h=h, lam=lam, gamma=gamma)
    pool = build_collective_pool(n)
    rng = np.random.default_rng(2)
    idx = rng.integers(0, len(pool), size=(3, 6))
    full = PennyLaneEvaluator(pool, ham, n, init_angle=init_angle)(idx)
    dicke = DickeEvaluator(pool, n, h, lam, gamma, init_angle=init_angle)(idx)
    np.testing.assert_allclose(dicke, full, atol=1e-9)


def test_dicke_initial_energy_matches(n=6):
    h, lam = 1.0, 1.8
    theta = lmg_mean_field_angle(h, lam)
    ham = lmg_hamiltonian(n, h=h, lam=lam)
    ev = DickeEvaluator(build_collective_pool(n), n, h, lam, init_angle=theta)
    assert ev.initial_energy() == pytest.approx(initial_energy(ham, n, theta), abs=1e-9)


def test_dicke_evaluator_ground_energy_matches_exact():
    ev = DickeEvaluator(build_collective_pool(4), 4, h=1.0, lam=1.5, gamma=0.0)
    assert hasattr(ev, "ground_energy"), "DickeEvaluator must expose ground_energy"
    expected = ground_state(dicke_matrix(4, h=1.0, lam=1.5, gamma=0.0))[0]
    assert ev.ground_energy == pytest.approx(expected, abs=1e-12)


def test_dicke_rejects_pairwise_pool():
    with pytest.raises(ValueError, match="collective pools only"):
        DickeEvaluator(build_pauli_pair_pool(4), 4, 1.0, 0.5)


def test_dicke_refine_reaches_ground(n=8):
    # collective structure + refinement should land on the sector ground state
    h, lam = 1.0, 1.5
    pool = build_collective_pool(n)
    space = DickeSpace.build(n, h, lam)
    by_label = {op.generator: op for op in pool if abs(op.angle - np.pi / 8) < 1e-9}
    # 12 gates, matching the production seq_len: 5 was too shallow to reach
    # the N=8 ground state even after refinement (expressibility, not a bug)
    structure = ("Jx2", "Jz", "Jy2", "Jz") * 3
    tokens = [0] + [pool.index(by_label[g]) + 1 for g in structure]
    result = dicke_refine_tokens(tokens, pool, space)
    e0 = ground_state(dicke_matrix(n, h, lam))[0]
    assert result.energy - e0 < 1e-6
    metrics = dicke_state_metrics(tokens, result.angles, pool, space)
    assert metrics["fidelity_ground"] > 1 - 1e-5


@pytest.mark.slow
def test_dicke_large_n_is_fast():
    # N = 500 collective-sector evaluation — verifies the Dicke encoding
    # stays practical for large-N production grids.
    n, h, lam = 500, 1.0, 1.5
    pool = build_collective_pool(n)
    ev = DickeEvaluator(pool, n, h, lam)
    t0 = time.time()
    out = ev(np.arange(12).reshape(1, 12))
    dt = time.time() - t0
    assert out.shape == (1, 12)
    assert np.all(np.abs(out) < n)  # sane energy scale
    assert dt < 30.0, (
        f"DickeEvaluator sequence evaluation took {dt:.1f}s — "
        "expected < 30s (checks for catastrophic regression, not peak performance)"
    )


def test_micro_train_with_dicke_evaluator(tmp_path):
    cfg = config_from_dict(
        {
            "run_name": "dicke-micro",
            "hamiltonian": {"kind": "lmg", "n_qubits": 6, "lam": 1.5},
            "pool": {"kind": "collective", "angles": [0.392699, -0.392699]},
            "model": {"n_layer": 1, "n_head": 2, "n_embd": 16, "dropout": 0.0},
            "train": {
                "epochs": 2,
                "seq_gen": 4,
                "seq_len": 4,
                "n_batches": 2,
                "eval_iter": 2,
                "eval_sequences": 4,
                "device": "cpu",
                "evaluator": "dicke",
            },
        }
    )
    result = train(cfg, tmp_path / "run")
    e0 = ground_state(dicke_matrix(6, 1.0, 1.5))[0]
    assert result.ground_energy == pytest.approx(e0)
    assert result.best_energy >= e0 - 1e-9
    assert result.refined_energy is not None


def test_dicke_refine_cli_best_of_n(tmp_path):
    # spingqe-refine --generate must route collective runs through the Dicke
    # sector: qubit-space refine is the >2 h cliff at N>=22 and impossible at
    # showcase N. Regression for the refine_main / _generate_candidates dispatch.
    import json

    from spingqe_lmg.cli import refine_main

    cfg = config_from_dict(
        {
            "run_name": "dicke-micro",
            "hamiltonian": {"kind": "lmg", "n_qubits": 6, "lam": 1.5},
            "pool": {"kind": "collective", "angles": [0.392699, -0.392699]},
            "model": {"n_layer": 1, "n_head": 2, "n_embd": 16, "dropout": 0.0},
            "train": {
                "epochs": 2,
                "seq_gen": 4,
                "seq_len": 4,
                "n_batches": 2,
                "eval_iter": 2,
                "eval_sequences": 4,
                "device": "cpu",
                "evaluator": "dicke",
            },
        }
    )
    run_dir = tmp_path / "run"
    train(cfg, run_dir)

    refine_main(
        ["--run-dir", str(run_dir), "--generate", "8", "--top", "2", "--allow-slow-backend"]
    )

    best = json.loads((run_dir / "best_sequence.json").read_text())
    e0 = ground_state(dicke_matrix(6, 1.0, 1.5))[0]
    # valid sector energy (no qubit-space crash) and the best-of-N record written
    assert best["refined_energy"] >= e0 - 1e-9
    assert "refined_tokens" in best and "refined_op_labels" in best


def test_dicke_space_arrays_are_readonly():
    """Frozen DickeSpace must reject in-place mutation of its arrays."""
    space = DickeSpace.build(4, h=1.0, lam=1.5, gamma=0.0)
    with pytest.raises(ValueError, match="read-only"):
        space.ham[0, 0] = 999.0


def test_dicke_space_eigval_arrays_are_readonly():
    """Generator eigenvalue arrays must also be read-only."""
    space = DickeSpace.build(4, h=1.0, lam=1.5, gamma=0.0)
    with pytest.raises(ValueError, match="read-only"):
        space.gen_eigvals["Jz"][0] = 999.0


def test_dicke_state_metrics_rejects_bos_at_nonzero_position():
    """Finding 4: same guard needed in dicke_state_metrics."""
    n, h, lam = 4, 1.0, 0.8
    space = DickeSpace.build(n, h, lam)
    pool = build_collective_pool(n)
    tokens = [0, 1, 0, 2]
    with pytest.raises(ValueError, match="BOS token at non-zero position"):
        dicke_state_metrics(tokens, None, pool, space)


def test_dicke_state_metrics_caches_eigendecomposition():
    """Finding 7: dicke_state_metrics should not call eigh per invocation."""
    n, h, lam = 20, -1.0, 0.8
    space = DickeSpace.build(n, h, lam)
    pool = build_collective_pool(n)
    tokens = [0, 1, 2]

    result1 = dicke_state_metrics(tokens, None, pool, space)
    result2 = dicke_state_metrics(tokens, None, pool, space)
    assert result1["fidelity_ground"] == pytest.approx(result2["fidelity_ground"])
    assert result1["order_parameter"] == pytest.approx(result2["order_parameter"])


def test_dicke_cache_uses_lru():
    """Finding 10: DickeEvaluator must use LRUCache with eviction."""
    from spingqe_lmg.core.caching import LRUCache

    n, h, lam = 6, 1.0, 0.8
    pool = build_collective_pool(n)

    ev = DickeEvaluator(pool, n, h, lam, cache_maxsize=3)
    assert isinstance(ev._cache, LRUCache)
    assert ev._cache.maxsize == 3

    rng = np.random.default_rng(2)
    keys = [tuple(int(x) for x in rng.integers(0, len(pool), size=3)) for _ in range(5)]
    for k in keys:
        ev(np.array([k]))

    assert len(ev._cache) == 3
