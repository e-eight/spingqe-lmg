"""Minimal walkthrough of the spingqe_lmg Python API.

Builds a tiny LMG Hamiltonian and operator pool, evaluates a hand-picked
gate sequence's energy directly with an evaluator (no training, no config
file), and checks the result against exact diagonalization.

This is the library's building blocks in isolation. For the full generative
pipeline (train a model to discover good sequences, then refine them), see
the CLI quickstart in the README, or run:
    spingqe-train --config configs/quickstart.toml --out-dir results/quickstart

Runs in seconds on a login node (CPU only, N=4, no GPU/SLURM needed):
    python examples/quickstart.py
"""

import numpy as np

from spingqe_lmg.evaluators import IncrementalEvaluator
from spingqe_lmg.exact import dense_matrix, ground_state
from spingqe_lmg.hamiltonians import lmg_hamiltonian
from spingqe_lmg.pools import build_pauli_pair_pool

N_QUBITS = 4
H = 1.0
LAM = 1.5  # lam > h: broken-symmetry phase (LMG's QPT sits at lam = h)

# 1. Build the Hamiltonian: H = -(h/2) sum Z_i - (lam/4N) sum_{i!=j} X_i X_j
ham = lmg_hamiltonian(N_QUBITS, h=H, lam=LAM)

# 2. Build an operator pool: two-qubit PauliRot gates (ZZ/XX/YY) + single-Z,
#    all-to-all connectivity.
pool = build_pauli_pair_pool(N_QUBITS)
print(f"N={N_QUBITS} qubits, lam/h={LAM / H}, pool has {len(pool)} operators")

# 3. Construct an evaluator directly (no training loop): the incremental
#    full-Hilbert-space evaluator holds a 2^N statevector and steps it
#    gate-by-gate, computing one energy per prefix of the sequence.
evaluator = IncrementalEvaluator(pool, ham, N_QUBITS)
e_initial = evaluator.initial_energy()
print(f"Initial (all-|0>) state energy: {e_initial:.6f}")

# 4. Evaluate one 3-gate sequence — a batch of pool-index rows, shape (1, 3).
sequence = np.array([[0, 5, 10]])
energies = evaluator(sequence)
print(f"Energy after each of the 3 gates: {energies[0]}")

# 5. Cross-check against exact diagonalization of the same Hamiltonian.
exact_e0 = ground_state(dense_matrix(ham, N_QUBITS))[0]
print(f"Exact ground-state energy: {exact_e0:.6f}")
print(f"Gap remaining after 3 (unoptimized) gates: {energies[0][-1] - exact_e0:.6f}")
