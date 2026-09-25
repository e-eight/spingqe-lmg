import numpy as np
import pytest

from spingqe_lmg.enums import RefineGradient, RefineOptimizer
from spingqe_lmg.exact import dicke_matrix, ground_state
from spingqe_lmg.hamiltonians import lmg_hamiltonian
from spingqe_lmg.pools import build_collective_pool, build_pauli_pair_pool
from spingqe_lmg.refine import refine_angles, refine_tokens

GRADIENTS = [RefineGradient.ADJOINT, RefineGradient.FINITE_DIFF]


def test_refine_never_worsens():
    n = 3
    ham = lmg_hamiltonian(n, h=1.0, lam=0.8)
    pool = build_pauli_pair_pool(n)
    rng = np.random.default_rng(11)
    ops = [pool[i] for i in rng.integers(0, len(pool), size=5)]
    result = refine_angles(ops, ham, n)
    assert result.energy <= result.initial_energy + 1e-9


@pytest.mark.parametrize("gradient", GRADIENTS)
def test_refine_reaches_ground_n2(gradient):
    # In the even-parity {|00>, |11>} subspace, XX acts as sigma_x and Z(0) as
    # sigma_z, so XX-Z-XX rotations cover the whole subspace: refinement of that
    # structure must land on the exact ground state under either gradient.
    n, h, lam = 2, 1.0, 0.5
    ham = lmg_hamiltonian(n, h=h, lam=lam)
    pool = build_pauli_pair_pool(n, angles=(np.pi / 8,))
    by_label = {op.label: op for op in pool}
    ops = [
        by_label["XX(0,1,+0.3927)"],
        by_label["Z(0,+0.3927)"],
        by_label["XX(0,1,+0.3927)"],
    ]
    result = refine_angles(ops, ham, n, gradient=gradient)
    assert abs(result.energy - ground_state(dicke_matrix(n, h, lam))[0]) < 1e-6


def test_gradient_methods_agree():
    n, h, lam = 2, 1.0, 0.5
    ham = lmg_hamiltonian(n, h=h, lam=lam)
    pool = build_pauli_pair_pool(n, angles=(np.pi / 8,))
    by_label = {op.label: op for op in pool}
    ops = [by_label["XX(0,1,+0.3927)"], by_label["Z(0,+0.3927)"], by_label["XX(0,1,+0.3927)"]]
    adj = refine_angles(ops, ham, n, gradient=RefineGradient.ADJOINT)
    fd = refine_angles(ops, ham, n, gradient=RefineGradient.FINITE_DIFF)
    assert abs(adj.energy - fd.energy) < 1e-8
    assert adj.n_gradient_evaluations > 0
    assert fd.n_gradient_evaluations == 0


def test_adjoint_gradient_matches_central_differences():
    # the jacobian L-BFGS consumes must be the true derivative of the cost
    import pennylane as qml

    from spingqe_lmg.evaluators.pennylane import prepare_initial_state

    n = 3
    ham = lmg_hamiltonian(n, h=1.0, lam=1.2)
    pool = build_pauli_pair_pool(n, paulis=("ZZ", "XX", "YY", "YZ", "XY"))
    rng = np.random.default_rng(31)
    ops = [pool[i] for i in rng.integers(0, len(pool), size=5)]

    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, diff_method="adjoint")
    def cost(params):
        prepare_initial_state(n, 0.0)
        for op, theta in zip(ops, params, strict=True):
            op.gate(angle=theta)
        return qml.expval(ham)

    x = rng.normal(scale=0.4, size=len(ops))
    adjoint_grad = np.asarray(qml.grad(cost)(qml.numpy.array(x, requires_grad=True)))
    eps = 1e-6
    for k in range(len(x)):
        step = np.zeros_like(x)
        step[k] = eps
        central = (float(cost(x + step)) - float(cost(x - step))) / (2 * eps)
        assert adjoint_grad[k] == pytest.approx(central, abs=1e-6)


def test_refine_collective_ops():
    n = 4
    ham = lmg_hamiltonian(n, h=1.0, lam=1.5)
    pool = build_collective_pool(n, angles=(0.3,))
    adj = refine_angles(list(pool), ham, n, gradient=RefineGradient.ADJOINT)
    fd = refine_angles(list(pool), ham, n, gradient=RefineGradient.FINITE_DIFF)
    assert adj.energy <= adj.initial_energy + 1e-9
    assert adj.energy <= fd.energy + 1e-8


def test_refine_rejects_unknown_gradient():
    n = 2
    ham = lmg_hamiltonian(n, h=1.0, lam=0.5)
    pool = build_pauli_pair_pool(n, angles=(0.4,))
    with pytest.raises((TypeError, ValueError)):
        refine_angles(pool[:2], ham, n, gradient="spsa")  # type: ignore


def test_refine_tokens_gradient_dispatch():
    n = 2
    ham = lmg_hamiltonian(n, h=1.0, lam=0.5)
    pool = build_pauli_pair_pool(n, angles=(0.4,))
    tokens = [0, 1, 2]
    adj = refine_tokens(tokens, pool, ham, n, gradient=RefineGradient.ADJOINT)
    fd = refine_tokens(tokens, pool, ham, n, gradient=RefineGradient.FINITE_DIFF)
    assert adj.n_gradient_evaluations > 0
    assert fd.n_gradient_evaluations == 0
    assert abs(adj.energy - fd.energy) < 1e-8


def test_refine_tokens_strips_bos():
    n = 2
    ham = lmg_hamiltonian(n, h=1.0, lam=0.5)
    pool = build_pauli_pair_pool(n, angles=(0.4,))
    tokens = [0, 1, 2]  # BOS + two pool ops
    result = refine_tokens(tokens, pool, ham, n)
    assert len(result.angles) == 2


def test_refine_tokens_respects_shots():
    n, h, lam = 2, 1.0, 0.5
    ham = lmg_hamiltonian(n, h=h, lam=lam)
    pool = build_pauli_pair_pool(n, angles=(np.pi / 8,))
    by_label = {op.label: op for op in pool}
    tokens = [0]
    for label in ["XX(0,1,+0.3927)", "Z(0,+0.3927)", "XX(0,1,+0.3927)"]:
        tokens.append(pool.index(by_label[label]) + 1)

    result = refine_tokens(
        tokens, pool, ham, n, shots=1000, seed=42, optimizer=RefineOptimizer.SPSA
    )
    assert result.energy <= result.initial_energy + 1e-6


def test_cli_refine_forwards_shots_seed_optimizer(tmp_path):
    import json
    from unittest.mock import patch

    from spingqe_lmg.cli import refine_main
    from spingqe_lmg.config import config_from_dict, config_to_dict

    cfg_dict = {
        "hamiltonian": {"kind": "lmg", "n_qubits": 4, "h": 1.0, "lam": 0.5, "gamma": 0.0},
        "train": {
            "evaluator": "incremental",
            "seed": 42,
            "seq_len": 6,
            "pennylane": {
                "device": "default.qubit",
                "shots": 100,
                "refine_shots": 100,
                "refine_optimizer": "cobyla",
            },
        },
        "pool": {"kind": "pauli_pair"},
        "model": {},
    }
    cfg = config_from_dict(cfg_dict)

    rd = tmp_path
    rd.joinpath("config.json").write_text(json.dumps(config_to_dict(cfg)))
    best = {"tokens": [0, 1, 2], "energy": -10.0, "ground_energy": -9.5}
    rd.joinpath("best_sequence.json").write_text(json.dumps(best))

    with patch("spingqe_lmg.evaluators.statevec.refine_tokens") as mock_refine:
        mock_refine.return_value.energy = -10.1
        mock_refine.return_value.angles = [0.1, 0.2]
        mock_refine.return_value.initial_energy = -10.0
        mock_refine.return_value.n_evaluations = 1
        refine_main(["--run-dir", str(rd), "--allow-slow-backend"])
        call_kwargs = mock_refine.call_args.kwargs
        assert call_kwargs["shots"] == 100
        assert call_kwargs["seed"] == 42
        assert call_kwargs["optimizer"] == RefineOptimizer.COBYLA


def test_refine_tokens_rejects_bos_at_nonzero_position():
    """Finding 4: BOS token (0) at non-zero position gives pool[-1] silently."""
    n, h, lam = 3, 1.0, 0.8
    ham = lmg_hamiltonian(n, h=h, lam=lam)
    pool = build_pauli_pair_pool(n)
    # tokens[2] is 0 (BOS) at a non-zero position
    tokens = [0, 1, 0, 3]
    with pytest.raises(ValueError, match="BOS token at non-zero position"):
        refine_tokens(tokens, pool, ham, n)
