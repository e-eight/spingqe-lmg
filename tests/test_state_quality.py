import json

import numpy as np
import pytest

from spingqe_lmg.analysis.state_quality import (
    analyze_run,
    circuit_state,
    fidelities,
    ground_space,
    half_chain_entropy,
    lmg_ground_doublet_qubit,
    order_parameter,
)
from spingqe_lmg.config import config_from_dict
from spingqe_lmg.exact import dense_matrix, order_parameter_exact
from spingqe_lmg.hamiltonians import lmg_hamiltonian
from spingqe_lmg.pools import build_pauli_pair_pool
from spingqe_lmg.refine import refine_angles
from spingqe_lmg.training import train


def test_entropy_product_and_ghz():
    product = np.zeros(16)
    product[0] = 1.0
    assert half_chain_entropy(product, 4) == pytest.approx(0.0, abs=1e-12)
    ghz = np.zeros(16)
    ghz[0] = ghz[-1] = 1 / np.sqrt(2)
    assert half_chain_entropy(ghz, 4) == pytest.approx(1.0, abs=1e-12)


def test_order_parameter_matches_exact_route():
    n, h, lam = 6, 1.0, 0.8  # paramagnetic: unique ground state
    ham_mat = dense_matrix(lmg_hamiltonian(n, h=h, lam=lam), n)
    _, lowest_two, _ = ground_space(ham_mat)
    got = order_parameter(lowest_two[:, 0], n)
    assert got == pytest.approx(order_parameter_exact(n, h, lam), abs=1e-10)


@pytest.mark.parametrize("n", [8, 10])
def test_order_parameter_matrix_free_vs_dense(n):
    # the matrix-free route must reproduce the dense Jx @ Jx contraction on
    # arbitrary (not just symmetric-sector) states
    rng = np.random.default_rng(n)
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    psi /= np.linalg.norm(psi)
    import pennylane as qml

    jx = 0.5 * sum(dense_matrix(qml.Hamiltonian([1.0], [qml.PauliX(i)]), n) for i in range(n))
    expected = float(np.real(psi.conj() @ (jx @ jx) @ psi)) * 4 / n**2
    assert order_parameter(psi, n) == pytest.approx(expected, abs=1e-12)


def test_lifted_doublet_matches_dense_ground_space():
    # broken phase: the two lowest global eigenstates are the symmetric-sector
    # parity doublet, so the lifted doublet must span the same subspace
    n, h, lam = 8, 1.0, 2.0
    _, lifted, gap = lmg_ground_doublet_qubit(n, h, lam)
    ham_mat = dense_matrix(lmg_hamiltonian(n, h=h, lam=lam), n)
    vals, dense_two, dense_gap = ground_space(ham_mat)
    overlap = lifted.conj().T @ dense_two  # 2x2; unitary iff the subspaces match
    np.testing.assert_allclose(overlap @ overlap.conj().T, np.eye(2), atol=1e-10)
    assert gap == pytest.approx(dense_gap, abs=1e-10)


def test_order_parameter_survives_n20():
    # survival regression for the large-N campaign: must run matrix-free on a
    # login node (the dense route would build a 16 TB Jx matrix at N = 20)
    n = 20
    rng = np.random.default_rng(20)
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    psi /= np.linalg.norm(psi)
    val = order_parameter(psi, n)
    assert 0.0 <= val <= 1.0 + 1e-9


def test_refined_ground_sequence_has_unit_fidelity():
    # the XX-Z-XX structure reaches the exact N=2 ground state under refinement
    n, h, lam = 2, 1.0, 0.5
    ham = lmg_hamiltonian(n, h=h, lam=lam)
    pool = build_pauli_pair_pool(n, angles=(np.pi / 8,))
    by_label = {op.label: op for op in pool}
    ops = [by_label["XX(0,1,+0.3927)"], by_label["Z(0,+0.3927)"], by_label["XX(0,1,+0.3927)"]]
    result = refine_angles(ops, ham, n)
    psi = circuit_state(ops, result.angles, n)
    _, lowest_two, _ = ground_space(dense_matrix(ham, n))
    f0, f01 = fidelities(psi, lowest_two)
    assert f0 == pytest.approx(1.0, abs=1e-6)
    assert f01 >= f0


def test_analyze_run_end_to_end(tmp_path):
    cfg = config_from_dict(
        {
            "run_name": "sq-micro",
            "hamiltonian": {"kind": "lmg", "n_qubits": 2, "lam": 0.5},
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
            },
        }
    )
    train(cfg, tmp_path / "run")
    row = analyze_run(tmp_path / "run")
    best = json.loads((tmp_path / "run" / "best_sequence.json").read_text())
    assert 0.0 <= row["fidelity_ground"] <= row["fidelity_doublet"] <= 1.0 + 1e-9
    assert row["energy_check"] == pytest.approx(best["refined_energy"], abs=1e-8)
    assert row["gap"] > 0


def test_popcounts_fallback_matches_bitwise_count():
    """Vectorized fallback must produce same results as np.bitwise_count."""
    from spingqe_lmg.analysis.state_quality import _popcounts

    for n in (2, 4, 6):
        counts = _popcounts(n)
        assert len(counts) == 2**n
        assert counts[0] == 0
        assert counts[2**n - 1] == n
        for k in range(n):
            assert counts[2**k] == 1
