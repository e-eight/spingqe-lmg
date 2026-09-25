"""State-quality metrics for completed runs.

Energy error alone can overstate or understate a variational state; these
metrics characterize whether the (refined) GQE circuit actually prepares
the right state:

- fidelity with the exact ground state, and with the ground *doublet*
  subspace (in the broken phase the finite-N spectrum has a near-degenerate
  parity pair),
- the QPT order parameter <(2 Jx / N)^2>,
- half-chain von Neumann entanglement entropy.

For the LMG model the exact references come from the (N+1)-dim Dicke sector
(eigenvectors lifted to the qubit basis), and the order parameter is computed
matrix-free — no dense 2^N x 2^N object is built. The dense-diagonalization
path remains for heisenberg_xxz and as a small-N cross-check in the tests.
"""

import json
from math import comb
from pathlib import Path

import numpy as np
import pennylane as qml

from spingqe_lmg.circuits import prepare_initial_state
from spingqe_lmg.config import config_from_dict
from spingqe_lmg.core.statevec_ops import order_param_sq_jx
from spingqe_lmg.enums import HamiltonianKind
from spingqe_lmg.exact import dense_matrix, dicke_matrix, order_parameter_exact
from spingqe_lmg.hamiltonians import build_hamiltonian
from spingqe_lmg.pools import PoolOp, build_pool
from spingqe_lmg.training import initial_angle_for


def circuit_state(
    ops: list[PoolOp],
    angles: list[float] | None,
    n_qubits: int,
    init_angle: float = 0.0,
) -> np.ndarray:
    """Statevector prepared by the gate sequence (refined angles optional)."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def state():
        prepare_initial_state(n_qubits, init_angle)
        thetas = angles if angles is not None else [None] * len(ops)
        for op, theta in zip(ops, thetas, strict=True):
            op.gate(angle=theta)
        return qml.state()

    return np.asarray(state())


def ground_space(ham_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(E values, two lowest eigenvectors, gap E1 - E0)."""
    vals, vecs = np.linalg.eigh(ham_matrix)
    return vals, vecs[:, :2], float(vals[1] - vals[0])


def fidelities(psi: np.ndarray, lowest_two: np.ndarray) -> tuple[float, float]:
    """(|<psi0|psi>|^2, fidelity with the span of the two lowest states)."""
    f0 = float(np.abs(lowest_two[:, 0].conj() @ psi) ** 2)
    f01 = f0 + float(np.abs(lowest_two[:, 1].conj() @ psi) ** 2)
    return f0, f01


def order_parameter(psi: np.ndarray, n_qubits: int) -> float:
    """<(2 Jx / N)^2> in the prepared state (matrix-free, O(N 2^N))."""
    return order_param_sq_jx(psi, n_qubits)


def _popcounts(n_qubits: int) -> np.ndarray:
    """Number of set bits for every computational-basis index 0 .. 2^N - 1."""
    idx = np.arange(2**n_qubits, dtype=np.uint64)
    try:
        return np.bitwise_count(idx).astype(np.intp)  # numpy >= 2.0
    except AttributeError:  # pragma: no cover - older numpy fallback
        byte_view = idx.view(np.uint8).reshape(-1, idx.itemsize)
        bits = np.unpackbits(byte_view, axis=1)
        return bits.sum(axis=1).astype(np.intp)


def lmg_ground_doublet_qubit(
    n_qubits: int, h: float, lam: float, gamma: float = 0.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """LMG sector eigenpairs lifted to the qubit basis: (E values, two lowest
    eigenvectors as 2^N qubit-basis columns, sector gap E1 - E0).

    The LMG ground doublet lives in the symmetric j = N/2 sector, so this is
    exact: diagonalize the (N+1)-dim Dicke matrix and lift each |j, m> with
    amplitude c_m / sqrt(C(N, k)) on every bitstring with k = popcount set
    bits (m = N/2 - k, i.e. Dicke row index = k). O(2^N) memory — never the
    dense 2^N x 2^N Hamiltonian.
    """
    vals, vecs = np.linalg.eigh(dicke_matrix(n_qubits, h, lam, gamma))
    k = _popcounts(n_qubits)
    weights = 1.0 / np.sqrt([comb(n_qubits, m) for m in range(n_qubits + 1)])
    lifted = vecs[k, :2] * weights[k][:, None]
    return vals, lifted, float(vals[1] - vals[0])


def state_energy(psi: np.ndarray, ham: qml.Hamiltonian, n_qubits: int) -> float:
    """<psi|H|psi> via a device expectation value.

    Never via qml.matrix(ham): its dense 2^N x 2^N intermediates are the OOM
    class that exceeds available memory at N >= 16 (see energy.initial_energy).
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def expval():
        qml.StatePrep(psi, wires=range(n_qubits))
        return qml.expval(ham)

    return float(np.real(expval()))


def half_chain_entropy(psi: np.ndarray, n_qubits: int) -> float:
    """Von Neumann entropy (base 2) of the first floor(N/2) qubits."""
    n_a = n_qubits // 2
    mat = psi.reshape(2**n_a, 2 ** (n_qubits - n_a))
    s = np.linalg.svd(mat, compute_uv=False)
    p = s**2
    p = p[p > 1e-14]
    return float(-np.sum(p * np.log2(p)))


def analyze_run(run_dir: str | Path) -> dict:
    """State-quality metrics for one completed run (uses the refined sequence)."""
    run_dir = Path(run_dir)
    cfg_data = json.loads((run_dir / "config.json").read_text())
    for extra in ("vocab_size", "pool_size"):
        cfg_data.pop(extra, None)
    cfg = config_from_dict(cfg_data)
    best = json.loads((run_dir / "best_sequence.json").read_text())

    n = cfg.hamiltonian.n_qubits
    pool = build_pool(cfg.pool, n)
    tokens = best.get("refined_tokens", best["tokens"])
    ops = [pool[t - 1] for t in tokens[1:]]
    angles = best.get("refined_angles")
    psi = circuit_state(ops, angles, n, init_angle=initial_angle_for(cfg))

    hc = cfg.hamiltonian
    if hc.kind is HamiltonianKind.LMG:
        # Dicke-sector references — exact for the LMG ground doublet, and the
        # only route that survives large N (no dense 2^N x 2^N objects).
        _, lowest_two, gap = lmg_ground_doublet_qubit(n, hc.h, hc.lam, hc.gamma)
        order_exact = order_parameter_exact(n, hc.h, hc.lam, hc.gamma)
        energy_check = state_energy(psi, build_hamiltonian(hc), n)
    else:
        ham_mat = dense_matrix(build_hamiltonian(hc), n)
        _, lowest_two, gap = ground_space(ham_mat)
        order_exact = order_parameter(lowest_two[:, 0], n)
        energy_check = float(np.real(psi.conj() @ ham_mat @ psi))
    f0, f01 = fidelities(psi, lowest_two)
    return {
        "run": run_dir.name,
        "fidelity_ground": f0,
        "fidelity_doublet": f01,
        "gap": gap,
        "order_parameter": order_parameter(psi, n),
        "order_parameter_exact": order_exact,
        "entropy_half_chain": half_chain_entropy(psi, n),
        "energy_check": energy_check,
    }


def analyze_sweep(sweep_dir: str | Path) -> "list[dict]":
    """State-quality rows for every completed run in a sweep directory."""
    sweep_dir = Path(sweep_dir)
    rows = []
    for run_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        if (run_dir / "best_sequence.json").exists() and (run_dir / "metadata.json").exists():
            rows.append(analyze_run(run_dir))
    return rows
