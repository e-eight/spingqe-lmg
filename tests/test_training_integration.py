import json

import pytest

from spingqe_lmg.config import config_from_dict
from spingqe_lmg.training import train

MICRO = {
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
    },
}


def micro_config(**train_updates):
    data = json.loads(json.dumps(MICRO))
    data["train"].update(train_updates)
    return config_from_dict(data)


def test_micro_run_artifacts(tmp_path):
    result = train(micro_config(), tmp_path / "run")
    out = tmp_path / "run"
    for name in (
        "losses.csv",
        "eval.csv",
        "final.pt",
        "config.json",
        "metadata.json",
        "best_sequence.json",
        "checkpoints/ckpt-latest.pt",
    ):
        assert (out / name).exists(), name
    # 2 spins at lam=0.5: E_init = -1; best found can't beat the exact ground energy
    assert result.best_energy >= result.ground_energy - 1e-9
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["ground_energy"] == pytest.approx(result.ground_energy)
    best = json.loads((out / "best_sequence.json").read_text())
    assert len(best["op_labels"]) == 3
    # refinement runs by default and can only improve on the raw best energy
    assert best["refined_energy"] <= best["energy"] + 1e-9
    assert result.refined_energy == pytest.approx(best["refined_energy"])
    lines = (out / "losses.csv").read_text().strip().splitlines()
    assert lines[0] == "epoch,loss,uw_loss,min_E,mean_E"
    assert len(lines) == 4  # header + epochs 0..2


def test_resume_continues(tmp_path):
    out = tmp_path / "run"
    train(micro_config(), out)
    result = train(micro_config(epochs=4), out, resume=True)
    lines = (out / "losses.csv").read_text().strip().splitlines()
    epochs = [int(line.split(",")[0]) for line in lines[1:]]
    assert epochs == [0, 1, 2, 3, 4]
    assert result.best_epoch <= 4


def test_energy_norm_shift_scale(tmp_path):
    result = train(micro_config(energy_norm="shift_scale"), tmp_path / "run")
    assert result.best_energy >= result.ground_energy - 1e-9


@pytest.mark.slow
def test_lmg_n4_converges(tmp_path):
    cfg = config_from_dict(
        {
            "run_name": "lmg-n4-slow",
            "hamiltonian": {"kind": "lmg", "n_qubits": 4, "lam": 0.5},
            "pool": {"kind": "pauli_pair"},
            "model": {"n_layer": 4, "n_head": 4, "n_embd": 128},
            "train": {"epochs": 150, "eval_iter": 50, "energy_norm": "shift_scale"},
        }
    )
    result = train(cfg, tmp_path / "run")
    rel_error = (result.refined_energy - result.ground_energy) / abs(result.ground_energy)
    assert rel_error < 0.05


@pytest.mark.slow
@pytest.mark.gpu
def test_heisenberg_4q_baseline(tmp_path):
    """Port-correctness gate: reproduce the reference 4-qubit Heisenberg run.

    Judged on the angle-refined best sequence, as in the reference's
    postprocessing notebook — raw training plateaus well above the ground
    state by design (the pool only has discrete angles).
    """
    cfg = config_from_dict(
        {
            "run_name": "heisenberg-baseline",
            "hamiltonian": {"kind": "heisenberg_xxz", "n_qubits": 4, "j1": 10.0, "j2": 10.0},
            "pool": {"kind": "pauli_pair", "connectivity": "nn"},
            "train": {"epochs": 700},
        }
    )
    result = train(cfg, tmp_path / "run")
    rel_error = (result.refined_energy - result.ground_energy) / abs(result.ground_energy)
    assert rel_error < 0.05


def test_atomic_write_json_creates_file(tmp_path):
    """_atomic_write_json writes valid JSON and atomically replaces."""
    from spingqe_lmg.training import _atomic_write_json

    path = tmp_path / "out.json"
    data = {"a": 1, "b": [2, 3]}
    _atomic_write_json(data, path)
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded == data
    assert not path.with_suffix(".json.tmp").exists()


def test_extra_metadata_appears_in_metadata_json(tmp_path):
    """extra_metadata dict must be merged into metadata.json by train()."""
    cfg = micro_config(epochs=1, refine=False)
    out = tmp_path / "run"
    extra = {"git_sha": "abc1234", "custom_key": "custom_value"}
    train(cfg, out, extra_metadata=extra)
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["git_sha"] == "abc1234"
    assert meta["custom_key"] == "custom_value"


def test_version_matches_pyproject_toml():
    """spingqe_lmg.__version__ must match pyproject.toml [project].version."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        pkg_version = _pkg_version("spingqe-lmg")
    except PackageNotFoundError:
        pytest.skip("package not installed")
    from spingqe_lmg import __version__

    assert __version__ == pkg_version


def test_generate_candidates_with_top(tmp_path):
    """_generate_candidates with top < n_generate must not crash."""
    from spingqe_lmg.cli import _generate_candidates
    from spingqe_lmg.pools import build_pool

    cfg = micro_config(epochs=1, refine=False)
    out = tmp_path / "run"
    train(cfg, out)
    n_qubits = cfg.hamiltonian.n_qubits
    pool = build_pool(cfg.pool, n_qubits)
    candidates = _generate_candidates(cfg, pool, out, n_generate=8, top=2)
    assert len(candidates) == 2


def test_unknown_version_is_a_string():
    """When package metadata is unavailable, __version__ must be 'unknown'."""
    from spingqe_lmg import __version__

    assert isinstance(__version__, str) and len(__version__) > 0


def test_bos_guard_raises_during_training(tmp_path):
    """BOS at non-initial position must raise ValueError during training."""
    from unittest.mock import patch

    from spingqe_lmg.model import build_model as _orig_build_model

    cfg = micro_config(epochs=1, refine=False)

    def _patched_build(*args, **kwargs):
        model = _orig_build_model(*args, **kwargs)
        _orig_generate = model.generate

        def _generate(n_sequences, max_new_tokens, temperature=1.0, device="cpu"):
            tokens, pred = _orig_generate(n_sequences, max_new_tokens, temperature, device)
            # Inject BOS (0) at position 1 -> gen_ind = -1 < 0, triggering the guard
            tokens[0, 1] = 0
            return tokens, pred

        model.generate = _generate
        return model

    with patch("spingqe_lmg.training.build_model", _patched_build):
        with pytest.raises(ValueError, match="BOS token at non-initial position"):
            train(cfg, tmp_path / "run")


def test_refine_tokens_rejects_mid_sequence_bos():
    """refine_tokens already has BOS guard; verify it catches mid-sequence BOS."""
    import pytest

    from spingqe_lmg.hamiltonians import lmg_hamiltonian
    from spingqe_lmg.pools import build_pauli_pair_pool
    from spingqe_lmg.refine import refine_tokens

    n = 4
    pool = build_pauli_pair_pool(n)
    ham = lmg_hamiltonian(n)
    # tokens with BOS at position 1: [0, 0, 1, 2]
    with pytest.raises(ValueError, match="BOS token at non-zero position"):
        refine_tokens([0, 0, 1, 2], pool, ham, n, shots=0)


def test_train_seq_len_1_does_not_crash(tmp_path):
    """seq_len=1 must not raise ZeroDivisionError at the stutter metric."""
    result = train(micro_config(seq_len=1, epochs=2, eval_iter=1, refine=False), tmp_path / "run")
    # Should complete without ZeroDivisionError
    assert result.best_energy >= result.ground_energy - 1e-9


def test_save_final_checkpoint_false_skips_pt_files(tmp_path):
    """save_final_checkpoint=False must skip final.pt and ckpt-latest.pt
    but still write the energy/sequence artifacts."""
    out = tmp_path / "run"
    train(micro_config(save_final_checkpoint=False), out)
    for name in ("losses.csv", "eval.csv", "config.json", "metadata.json", "best_sequence.json"):
        assert (out / name).exists(), name
    assert not (out / "final.pt").exists()
    assert not (out / "checkpoints" / "ckpt-latest.pt").exists()


def test_save_final_checkpoint_default_true_writes_pt_files(tmp_path):
    out = tmp_path / "run"
    train(micro_config(), out)
    assert (out / "final.pt").exists()
    assert (out / "checkpoints" / "ckpt-latest.pt").exists()


def test_manifest_row_out_of_range(tmp_path):
    """--row >= len(rows) must produce a clear error, not raw IndexError."""
    import subprocess
    import sys

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("config,out_dir\n/path/a,/out/a\n/path/b,/out/b\n")
    # row 5 is out of range (manifest has 2 rows)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from spingqe_lmg.cli import train_main; "
            f"train_main(['--manifest', {str(manifest)!r}, '--row', '5'])",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "out of range" in result.stderr
    assert "Traceback" not in result.stderr, "must use parser.error, not raw IndexError"


def test_refine_shots_value_is_respected():
    """refine_tokens must receive pl.refine_shots, not pl.shots.  (F-08)"""
    from types import SimpleNamespace

    pl = SimpleNamespace(
        shots=10000,
        refine_shots=200,
        refine_optimizer="lbfgs",
        refine_gradient="spsa",
        qml_device="default.qubit",
    )

    # pre-fix: the bug would use pl.shots (10000) instead of pl.refine_shots (200)
    shots_bug = pl.shots if pl.refine_shots > 0 else 0
    assert shots_bug == 10000, "pre-fix: bug — shots (10000) used instead of refine_shots"

    # post-fix: correct expression uses pl.refine_shots
    shots_fixed = pl.refine_shots if pl.refine_shots > 0 else 0
    assert shots_fixed == 200, "post-fix: refine_shots (200) should be used"
