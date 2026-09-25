import json

import pytest

from spingqe_lmg.config import (
    apply_overrides,
    config_from_dict,
    config_to_dict,
    load_config,
)
from spingqe_lmg.enums import (
    Connectivity,
    DeviceKind,
    HamiltonianKind,
    PoolKind,
    RefineGradient,
    RefineOptimizer,
)


def test_defaults_are_reference_hyperparams():
    cfg = config_from_dict({})
    assert cfg.model.n_layer == 12
    assert cfg.model.n_head == 8
    assert cfg.model.n_embd == 512
    assert cfg.train.seq_len == 12
    assert cfg.train.temperature == 0.5
    assert cfg.train.beta == 0.3
    assert cfg.train.learning_rate == 5e-5
    assert len(cfg.pool.angles) == 10


def test_toml_roundtrip(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text(
        """
run_name = "test"
[hamiltonian]
kind = "lmg"
n_qubits = 6
lam = 1.5
[pool]
kind = "collective"
[train]
epochs = 10
"""
    )
    cfg = load_config(path)
    assert cfg.run_name == "test"
    assert cfg.hamiltonian.n_qubits == 6
    assert cfg.hamiltonian.lam == 1.5
    assert cfg.pool.kind is PoolKind.COLLECTIVE
    assert cfg.train.epochs == 10
    # untouched sections keep defaults
    assert cfg.model.n_layer == 12


def test_load_config_json_roundtrips_run_config(tmp_path):
    """A run directory's config.json reloads to the config that wrote it."""
    cfg = config_from_dict(
        {"hamiltonian": {"n_qubits": 6, "lam": 1.5}, "pool": {"kind": "collective"}}
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**config_to_dict(cfg), "vocab_size": 31, "pool_size": 30}))
    loaded = load_config(path, ["train.epochs=3"])
    assert loaded.hamiltonian.n_qubits == 6
    assert loaded.pool.kind is PoolKind.COLLECTIVE
    assert loaded.train.epochs == 3
    assert config_to_dict(loaded) == config_to_dict(cfg) | {
        "train": config_to_dict(cfg)["train"] | {"epochs": 3}
    }


def test_legacy_train_keys_move_into_pennylane():
    """Old run configs kept the PennyLane options directly under [train]."""
    cfg = config_from_dict(
        {
            "train": {
                "evaluator": "dicke",
                "qml_device": "lightning.qubit",
                "shots": 0,
                "refine_shots": False,
                "refine_optimizer": "spsa",
                "refine_gradient": "adjoint",
            }
        }
    )
    assert cfg.train.pennylane.device == "lightning.qubit"
    assert cfg.train.pennylane.refine_shots == 0
    assert cfg.train.pennylane.refine_optimizer is RefineOptimizer.SPSA


def test_overrides():
    data = apply_overrides(
        {},
        [
            "hamiltonian.lam=1.25",
            "train.epochs=5",
            "pool.kind=collective",
            'pool.connectivity="nn"',
            "train.device=cpu",
        ],
    )
    cfg = config_from_dict(data)
    assert cfg.hamiltonian.lam == 1.25
    assert cfg.train.epochs == 5
    assert cfg.pool.kind is PoolKind.COLLECTIVE
    assert cfg.pool.connectivity is Connectivity.NN
    assert cfg.train.device is DeviceKind.CPU


def test_option_typo_fails_at_parse_time():
    with pytest.raises(ValueError, match="PoolKind.*collectiv"):
        config_from_dict({"pool": {"kind": "collectiv"}})
    with pytest.raises(ValueError, match="RefineOptimizer.*qnspsa"):
        config_from_dict({"train": {"pennylane": {"refine_optimizer": "qnspsa"}}})


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="unknown"):
        config_from_dict({"hamiltonian": {"lambda": 1.0}})
    with pytest.raises(ValueError, match="unknown top-level"):
        config_from_dict({"typo_section": {}})


def test_sweep_table_is_ignored():
    cfg = config_from_dict({"hamiltonian": {"n_qubits": 6}, "sweep": {"seeds": [1, 2]}})
    assert cfg.hamiltonian.n_qubits == 6


def test_override_list_value_roundtrip():
    from spingqe_lmg.cli import _toml_literal

    literal = _toml_literal(["ZZ", "XX", "YY", "YZ", "XY"])
    cfg = config_from_dict(apply_overrides({}, [f"pool.paulis={literal}"]))
    assert cfg.pool.paulis == ("ZZ", "XX", "YY", "YZ", "XY")


def test_config_to_dict_roundtrip():
    cfg = config_from_dict({"hamiltonian": {"n_qubits": 8}})
    data = config_to_dict(cfg)
    roundtripped = config_from_dict(data)
    assert roundtripped.hamiltonian.kind is cfg.hamiltonian.kind
    assert roundtripped.hamiltonian.n_qubits == cfg.hamiltonian.n_qubits
    assert roundtripped.train.evaluator is cfg.train.evaluator


def test_angle_scale_roundtrip_and_override():
    cfg = config_from_dict({"pool": {"angle_scale": 8}})
    assert cfg.pool.angle_scale == 8
    data = config_to_dict(cfg)
    assert config_from_dict(data).pool.angle_scale == 8
    cfg = config_from_dict(apply_overrides({}, ["pool.angle_scale=8"]))
    assert cfg.pool.angle_scale == 8
    assert config_from_dict({}).pool.angle_scale == 0.0


def test_refine_gradient_option_validated():
    assert config_from_dict({}).train.pennylane is None
    cfg = config_from_dict({"train": {"pennylane": {"refine_gradient": "finite_diff"}}})
    assert cfg.train.pennylane is not None
    assert cfg.train.pennylane.refine_gradient is RefineGradient.FINITE_DIFF
    with pytest.raises(ValueError, match="RefineGradient.*spsa"):
        config_from_dict({"train": {"pennylane": {"refine_gradient": "spsa"}}})


def test_cross_constraints_caught_at_parse_time():
    with pytest.raises(ValueError, match="evaluator='dicke'.*lmg"):
        config_from_dict(
            {"hamiltonian": {"kind": "heisenberg_xxz"}, "train": {"evaluator": "dicke"}}
        )

    with pytest.raises(ValueError, match="init_state='ghz_x'"):
        config_from_dict({"train": {"evaluator": "dicke", "init_state": "ghz_x"}})

    with pytest.raises(ValueError, match="init_state='mean_field'"):
        config_from_dict(
            {"hamiltonian": {"kind": "heisenberg_xxz"}, "train": {"init_state": "mean_field"}}
        )

    with pytest.raises(ValueError, match="refine_shots>0 requires shots>0"):
        config_from_dict(
            {"train": {"evaluator": "pennylane", "pennylane": {"refine_shots": 100, "shots": 0}}}
        )


def test_enum_field_is_enum():
    cfg = config_from_dict({})
    assert cfg.hamiltonian.kind is HamiltonianKind.LMG
    assert cfg.train.evaluator.value == "pennylane"
    assert cfg.train.init_state.value == "zero"
    assert cfg.train.pennylane is None
    assert cfg.model.kind.value == "gpt"
    pl_cfg = config_from_dict({"train": {"pennylane": {}}})
    assert pl_cfg.train.pennylane is not None
    assert pl_cfg.train.pennylane.refine_gradient.value == "adjoint"
    assert pl_cfg.train.pennylane.refine_optimizer.value == "spsa"


def test_hamiltonian_kind_enum():
    cfg = config_from_dict({"hamiltonian": {"kind": "heisenberg_xxz"}})
    assert cfg.hamiltonian.kind is HamiltonianKind.HEISENBERG_XXZ


def test_pool_kind_enum():
    cfg = config_from_dict({"pool": {"kind": "collective"}})
    assert cfg.pool.kind is PoolKind.COLLECTIVE
    assert cfg.pool.connectivity is Connectivity.ALL


def test_evaluator_enum():
    cfg = config_from_dict({"train": {"evaluator": "incremental"}})
    assert cfg.train.evaluator.value == "incremental"


def test_override_parsing_preserves_strings():
    """--set train.refine_shots=true must stay 'true' (string), not become True."""
    from spingqe_lmg.config import _parse_override_value

    assert _parse_override_value("true") == "true"
    assert _parse_override_value("false") == "false"
    assert _parse_override_value("collective") == "collective"


def test_override_parsing_still_parses_lists():
    """Lists must still be parsed from TOML."""
    from spingqe_lmg.config import _parse_override_value

    result = _parse_override_value('["XY", "YX"]')
    assert isinstance(result, list)
    assert result == ["XY", "YX"]


def test_override_parsing_keeps_bools_as_strings():
    """'true'/'false' must stay strings for enum coercion elsewhere."""
    from spingqe_lmg.config import _parse_override_value

    assert _parse_override_value("true") == "true"
    assert _parse_override_value("false") == "false"


def test_apply_overrides_bool_field_becomes_real_bool():
    """--set train.refine=false must produce Python False, not a truthy string.

    Regression: _parse_override_value deliberately keeps 'true'/'false' as
    strings (for enum coercion), but plain bool fields (e.g. train.refine,
    train.save_final_checkpoint) were never coerced back, so the CLI override
    silently no-opped (a non-empty string is truthy).
    """
    from spingqe_lmg.config import config_from_dict

    data = {
        "hamiltonian": {"kind": "lmg", "n_qubits": 2, "lam": 0.5},
        "pool": {"kind": "pauli_pair"},
        "train": {"refine": "false"},
    }
    cfg = config_from_dict(data)
    assert cfg.train.refine is False


def test_build_section_coerces_enums_under_future_annotations():
    """_build_section must coerce enums even if annotations are strings (PEP 563)."""
    from spingqe_lmg.config import TrainConfig, _build_section

    raw = {
        "epochs": 1,
        "seq_len": 2,
        "temperature": 1.0,
        "beta": 0.0,
        "eval_sequences": 2,
        "eval_iter": 1,
        "checkpoint_every": 999,
        "seq_gen": 4,
        "n_batches": 1,
        "refine": False,
        "evaluator": "incremental",
    }
    cfg = _build_section(TrainConfig, raw)
    from spingqe_lmg.enums import EvaluatorKind

    assert cfg.evaluator is EvaluatorKind.INCREMENTAL


def test_override_parsing_coerces_numbers():
    """Numbers must still be parsed as int/float from TOML."""
    from spingqe_lmg.config import _parse_override_value

    assert _parse_override_value("5") == 5
    assert _parse_override_value("1.5") == 1.5


def test_hamiltonian_config_rejects_odd_n_qubits():
    """Parity-sector evaluators require even N; validate at config level."""
    with pytest.raises(ValueError, match="n_qubits must be a positive even integer"):
        config_from_dict({"hamiltonian": {"n_qubits": 3}})


def test_hamiltonian_config_rejects_n_qubits_lt_2():
    with pytest.raises(ValueError, match="n_qubits must be a positive even integer"):
        config_from_dict({"hamiltonian": {"n_qubits": 0}})
    with pytest.raises(ValueError, match="n_qubits must be a positive even integer"):
        config_from_dict({"hamiltonian": {"n_qubits": 1}})


def test_hamiltonian_config_accepts_even_n_qubits():
    """Even N >= 2 must be accepted."""
    cfg = config_from_dict({"hamiltonian": {"n_qubits": 4}})
    assert cfg.hamiltonian.n_qubits == 4
    cfg = config_from_dict({"hamiltonian": {"n_qubits": 6}})
    assert cfg.hamiltonian.n_qubits == 6


def test_markov_order_field():
    from spingqe_lmg.enums import ModelKind

    cfg = config_from_dict(
        {
            "run_name": "test",
            "hamiltonian": {"kind": "lmg", "n_qubits": 4, "h": 1.0, "lam": 0.8},
            "pool": {"kind": "pauli_pair"},
            "model": {"kind": "markov", "order": 2},
            "train": {
                "epochs": 1,
                "seq_len": 2,
                "temperature": 1.0,
                "beta": 0.0,
                "eval_sequences": 2,
                "eval_iter": 1,
                "checkpoint_every": 999,
                "refine": False,
            },
        }
    )
    assert cfg.model.kind is ModelKind.MARKOV
    assert cfg.model.order == 2


def test_markov1_toml_no_longer_valid():
    import pytest

    with pytest.raises((ValueError, KeyError)):
        config_from_dict(
            {
                "run_name": "test",
                "hamiltonian": {"kind": "lmg", "n_qubits": 4, "h": 1.0, "lam": 0.8},
                "pool": {"kind": "pauli_pair"},
                "model": {"kind": "markov1"},
                "train": {
                    "epochs": 1,
                    "seq_len": 2,
                    "temperature": 1.0,
                    "beta": 0.0,
                    "eval_sequences": 2,
                    "eval_iter": 1,
                    "checkpoint_every": 999,
                    "refine": False,
                },
            }
        )


def test_pennylane_config_nested():
    from spingqe_lmg.config import config_from_dict

    cfg = config_from_dict(
        {
            "run_name": "test",
            "hamiltonian": {"kind": "lmg", "n_qubits": 4, "h": 1.0, "lam": 0.8},
            "pool": {"kind": "pauli_pair"},
            "model": {"kind": "gpt", "n_layer": 1, "n_head": 1, "n_embd": 16},
            "train": {
                "epochs": 1,
                "seq_len": 2,
                "temperature": 1.0,
                "beta": 0.0,
                "eval_sequences": 2,
                "eval_iter": 1,
                "checkpoint_every": 999,
                "refine": False,
                "evaluator": "pennylane",
                "pennylane": {"device": "default.qubit", "shots": 500},
            },
        }
    )
    assert cfg.train.pennylane is not None
    assert cfg.train.pennylane.device == "default.qubit"
    assert cfg.train.pennylane.shots == 500


def test_non_pennylane_has_no_pennylane_config():
    from spingqe_lmg.config import config_from_dict

    cfg = config_from_dict(
        {
            "run_name": "test",
            "hamiltonian": {"kind": "lmg", "n_qubits": 4, "h": 1.0, "lam": 0.8},
            "pool": {"kind": "pauli_pair"},
            "model": {"kind": "gpt", "n_layer": 1, "n_head": 1, "n_embd": 16},
            "train": {
                "epochs": 1,
                "seq_len": 2,
                "temperature": 1.0,
                "beta": 0.0,
                "eval_sequences": 2,
                "eval_iter": 1,
                "checkpoint_every": 999,
                "refine": False,
                "evaluator": "incremental",
            },
        }
    )
    assert cfg.train.pennylane is None


@pytest.mark.parametrize("evaluator_kind", ["incremental_gpu", "parity_even_gpu"])
def test_build_evaluator_gpu_kinds_do_not_crash(evaluator_kind):
    """build_evaluator must handle GPU evaluator kinds and not raise ValueError."""
    pytest.importorskip("torch")
    from spingqe_lmg.evaluators import build_evaluator
    from spingqe_lmg.hamiltonians import build_hamiltonian
    from spingqe_lmg.pools import build_pool
    from spingqe_lmg.training import initial_angle_for

    cfg = config_from_dict(
        {
            "run_name": "test",
            "hamiltonian": {"kind": "lmg", "n_qubits": 4, "h": 1.0, "lam": 0.8},
            "pool": {"kind": "pauli_pair"},
            "model": {"kind": "gpt", "n_layer": 1, "n_head": 1, "n_embd": 16},
            "train": {
                "epochs": 1,
                "seq_len": 2,
                "temperature": 1.0,
                "beta": 0.0,
                "eval_sequences": 2,
                "eval_iter": 1,
                "checkpoint_every": 999,
                "refine": False,
                "evaluator": evaluator_kind,
            },
        }
    )
    try:
        pool = build_pool(cfg.pool, cfg.hamiltonian.n_qubits)
        ham = build_hamiltonian(cfg.hamiltonian)
        init_angle = initial_angle_for(cfg)
        evaluator, e0 = build_evaluator(
            cfg, pool, ham, init_angle=init_angle, n_qubits=cfg.hamiltonian.n_qubits
        )
    except ImportError:
        pytest.skip("GPU evaluator not importable on this node")
