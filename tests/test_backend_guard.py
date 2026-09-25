"""Backend guard: default.qubit must be refused on DeltaAI unless overridden.

Covers both CLI entry points: train_main (gated on train.refine) and
refine_main (always refines, so it must not be gated on that flag).
"""

import json

import pytest

from spingqe_lmg.cli import refine_main, train_main
from spingqe_lmg.cli.train import _check_backend_guard
from spingqe_lmg.config import config_from_dict
from spingqe_lmg.training import train

MICRO_TOML = """
run_name = "micro"

[hamiltonian]
kind = "lmg"
n_qubits = 2
lam = 0.5

[pool]
kind = "pauli_pair"
angles = [0.7853981633974483, -0.7853981633974483]

[model]
n_layer = 1
n_head = 2
n_embd = 16
dropout = 0.0

[train]
epochs = 2
seq_gen = 4
seq_len = 3
n_batches = 2
eval_iter = 2
eval_sequences = 4
checkpoint_every = 1
device = "cpu"
"""


@pytest.fixture
def micro_toml(tmp_path):
    path = tmp_path / "micro.toml"
    path.write_text(MICRO_TOML)
    return path


def _cfg(refine=True, device=None):
    data = {
        "hamiltonian": {"kind": "lmg", "n_qubits": 2, "lam": 0.5},
        "pool": {"kind": "pauli_pair"},
        "train": {"refine": refine},
    }
    if device is not None:
        data["train"]["pennylane"] = {"device": device}
    return config_from_dict(data)


# --- direct unit tests on _check_backend_guard ---


def test_guard_raises_on_default_qubit_when_refine_on():
    with pytest.raises(SystemExit, match="default.qubit"):
        _check_backend_guard(_cfg(refine=True), allow_slow_backend=False)


def test_guard_raises_on_unset_device_fallback():
    """No train.pennylane.device set at all -> falls back to default.qubit."""
    with pytest.raises(SystemExit, match="default.qubit"):
        _check_backend_guard(_cfg(refine=True, device=None), allow_slow_backend=False)


def test_guard_allows_lightning_qubit():
    _check_backend_guard(_cfg(refine=True, device="lightning.qubit"), allow_slow_backend=False)


def test_guard_allows_override_flag():
    _check_backend_guard(_cfg(refine=True), allow_slow_backend=True)


def test_guard_skips_when_refine_off():
    _check_backend_guard(_cfg(refine=False), allow_slow_backend=False)


def test_guard_assume_refine_ignores_train_refine_flag():
    """refine_main's case: train.refine=False must not bypass the guard."""
    with pytest.raises(SystemExit, match="default.qubit"):
        _check_backend_guard(_cfg(refine=False), allow_slow_backend=False, assume_refine=True)


def test_guard_assume_refine_still_respects_override():
    _check_backend_guard(_cfg(refine=False), allow_slow_backend=True, assume_refine=True)


# --- integration tests through the CLI entry points ---


def test_train_main_refuses_default_qubit(micro_toml, tmp_path):
    with pytest.raises(SystemExit, match="default.qubit"):
        train_main(["--config", str(micro_toml), "--out-dir", str(tmp_path / "run")])


def test_train_main_allow_slow_backend_proceeds(micro_toml, tmp_path):
    train_main(
        [
            "--config",
            str(micro_toml),
            "--out-dir",
            str(tmp_path / "run"),
            "--allow-slow-backend",
        ]
    )
    assert (tmp_path / "run" / "metadata.json").exists()


def test_train_main_refine_off_proceeds(micro_toml, tmp_path):
    train_main(
        [
            "--config",
            str(micro_toml),
            "--out-dir",
            str(tmp_path / "run"),
            "--set",
            "train.refine=false",
        ]
    )
    assert (tmp_path / "run" / "metadata.json").exists()


def _micro_run_config(**train_updates):
    data = {
        "run_name": "micro",
        "hamiltonian": {"kind": "lmg", "n_qubits": 2, "lam": 0.5},
        "pool": {"kind": "pauli_pair", "angles": [0.7853981633974483, -0.7853981633974483]},
        "model": {"n_layer": 1, "n_head": 2, "n_embd": 16, "dropout": 0.0},
        "train": {
            "epochs": 2,
            "seq_gen": 4,
            "seq_len": 3,
            "n_batches": 2,
            "eval_iter": 2,
            "eval_sequences": 4,
            "checkpoint_every": 1,
            "device": "cpu",
            **train_updates,
        },
    }
    return config_from_dict(data)


def test_refine_main_refuses_default_qubit(tmp_path):
    """refine_main always refines, so train.refine=False in the run's config
    must NOT bypass the guard (assume_refine=True)."""
    run_dir = tmp_path / "run"
    train(_micro_run_config(refine=False), run_dir)
    with pytest.raises(SystemExit, match="default.qubit"):
        refine_main(["--run-dir", str(run_dir)])


def test_refine_main_allow_slow_backend_proceeds(tmp_path):
    run_dir = tmp_path / "run"
    train(_micro_run_config(refine=False), run_dir)
    refine_main(["--run-dir", str(run_dir), "--allow-slow-backend"])
    best = json.loads((run_dir / "best_sequence.json").read_text())
    assert "refined_energy" in best
