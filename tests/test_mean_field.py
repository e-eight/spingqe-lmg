import json

import numpy as np
import pytest

from spingqe_lmg.config import config_from_dict
from spingqe_lmg.evaluators.pennylane import initial_energy, sequence_energies
from spingqe_lmg.exact import dicke_matrix, ground_state
from spingqe_lmg.hamiltonians import lmg_hamiltonian, lmg_mean_field_angle
from spingqe_lmg.pools import build_pauli_pair_pool
from spingqe_lmg.training import initial_angle_for, train


def test_mean_field_angle_values():
    assert lmg_mean_field_angle(1.0, 0.5) == 0.0
    assert lmg_mean_field_angle(1.0, 1.0) == 0.0
    assert lmg_mean_field_angle(1.0, 2.0) == pytest.approx(np.arccos(0.5))


@pytest.mark.parametrize("n", [2, 4, 8, 16])  # n=16 guards the dense-matrix OOM regression
@pytest.mark.parametrize("theta", [0.0, 0.4, 1.1])
def test_initial_energy_matches_product_state_formula(n, theta):
    h, lam = 1.0, 1.5
    ham = lmg_hamiltonian(n, h=h, lam=lam)
    # tilted product state: <Z_i> = cos(theta), <X_i X_j> = sin^2(theta)
    expected = -(n * h / 2) * np.cos(theta) - (lam * (n - 1) / 4) * np.sin(theta) ** 2
    assert initial_energy(ham, n, theta) == pytest.approx(expected, abs=1e-10)


def test_mean_field_beats_zero_basin_in_broken_phase():
    n, h, lam = 8, 1.0, 1.5
    ham = lmg_hamiltonian(n, h=h, lam=lam)
    theta = lmg_mean_field_angle(h, lam)
    e_mf = initial_energy(ham, n, theta)
    e_zero = initial_energy(ham, n, 0.0)
    e_exact = ground_state(dicke_matrix(n, h, lam))[0]
    assert e_mf < e_zero  # escapes the |0...0> basin before any gates
    assert e_mf > e_exact - 1e-12  # but still above the true ground energy


def test_sequence_energies_respect_init_angle():
    n, theta = 3, 0.9
    ham = lmg_hamiltonian(n, h=1.0, lam=1.5)
    pool = build_pauli_pair_pool(n, angles=(1e-12,), paulis=("ZZ",), include_single_z=False)
    es = sequence_energies((pool[0],), ham, n, init_angle=theta)
    assert es[0] == pytest.approx(initial_energy(ham, n, theta), abs=1e-9)


def test_initial_angle_for_config():
    cfg = config_from_dict(
        {"hamiltonian": {"kind": "lmg", "lam": 2.0}, "train": {"init_state": "mean_field"}}
    )
    assert initial_angle_for(cfg) == pytest.approx(np.arccos(0.5))
    with pytest.raises(ValueError, match="only defined for the LMG"):
        initial_angle_for(
            config_from_dict(
                {"hamiltonian": {"kind": "heisenberg_xxz"}, "train": {"init_state": "mean_field"}}
            )
        )


def test_micro_train_with_mean_field(tmp_path):
    cfg = config_from_dict(
        {
            "run_name": "micro-mf",
            "hamiltonian": {"kind": "lmg", "n_qubits": 2, "lam": 1.5},
            "pool": {"kind": "pauli_pair", "angles": [0.785, -0.785]},
            "model": {"n_layer": 1, "n_head": 2, "n_embd": 16, "dropout": 0.0},
            "train": {
                "epochs": 2,
                "seq_gen": 4,
                "seq_len": 3,
                "n_batches": 2,
                "eval_iter": 2,
                "eval_sequences": 4,
                "device": "cpu",
                "init_state": "mean_field",
            },
        }
    )
    result = train(cfg, tmp_path / "run")
    assert result.best_energy >= result.ground_energy - 1e-9
    meta = json.loads((tmp_path / "run" / "metadata.json").read_text())
    assert meta["init_state"] == "mean_field"
    assert meta["init_angle"] == pytest.approx(np.arccos(1.0 / 1.5))
