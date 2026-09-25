#!/usr/bin/env python
"""Continuous small-angle gradient probe: parameter-shift derivative at theta->0.

For each pool generator (collapsing the ten-angle discrete vocabulary to one
symmetric small angle), evaluates the exact energy at theta = +eps and
theta = -eps and reports the central-difference derivative dE/dtheta|_0.
Compares against the discrete descent test (manuscript Sec. IV, Table
"descent") reported in data/tables/descent-test-n16-lam2.0.json: does a continuous
small-angle probe find an improving direction exactly where the discrete
probe does, at the five tab:descent configurations?

Usage:
    python scripts/gradient-probe.py [--json data/tables/gradient-probe-n16-lam2.0.json]
"""

import argparse
import json

import numpy as np

from spingqe_lmg.evaluators.statevec import IncrementalEvaluator
from spingqe_lmg.hamiltonians import lmg_hamiltonian, lmg_mean_field_angle
from spingqe_lmg.pools import build_collective_pool, build_pauli_pair_pool

STD_PAULIS = ("ZZ", "XX", "YY")
EXT_PAULIS = ("ZZ", "XX", "YY", "YZ", "XY")

CONFIGS = {
    "std-pairwise-zero": ("std. pairwise, zero", STD_PAULIS, "zero"),
    "std-pairwise-mf": ("std. pairwise, mean field", STD_PAULIS, "mean_field"),
    "ext-pairwise-zero": ("ext. pairwise, zero", EXT_PAULIS, "zero"),
    "ext-pairwise-mf": ("ext. pairwise, mean field", EXT_PAULIS, "mean_field"),
    "collective-zero": ("collective, zero", None, "zero"),
}

N, H, LAM, EPS = 16, 1.0, 2.0, 1e-6
TOL = 1e-10  # same tolerance as the discrete descent test


def build_pool(paulis, n_qubits, angle):
    if paulis is None:
        return build_collective_pool(n_qubits, angles=(angle,))
    return build_pauli_pair_pool(n_qubits, angles=(angle,), paulis=paulis)


def gradient_probe(paulis, reference, n_qubits, h, lam, eps):
    ham = lmg_hamiltonian(n_qubits, h=h, lam=lam)
    init_angle = lmg_mean_field_angle(h, lam) if reference == "mean_field" else 0.0

    pool_plus = build_pool(paulis, n_qubits, eps)
    pool_minus = build_pool(paulis, n_qubits, -eps)
    gen_labels = [op.label for op in pool_plus]

    ev_plus = IncrementalEvaluator(pool_plus, ham, n_qubits, init_angle=init_angle)
    ev_minus = IncrementalEvaluator(pool_minus, ham, n_qubits, init_angle=init_angle)

    e_ref = ev_plus.initial_energy()
    n = len(pool_plus)
    idx = np.arange(n, dtype=np.int64).reshape(-1, 1)
    e_plus = ev_plus(idx)[:, 0]
    e_minus = ev_minus(idx)[:, 0]

    grad = (e_plus - e_minus) / (2 * eps)
    max_abs = np.max(np.abs(grad))
    improving = np.flatnonzero(np.abs(grad) > TOL / eps)  # nonzero grad -> an improving sign of theta exists
    best = int(np.argmax(np.abs(grad)))
    return {
        "reference": reference,
        "e_ref": float(e_ref),
        "n_generators": n,
        "max_abs_grad": float(max_abs),
        "best_generator": gen_labels[best],
        "best_grad": float(grad[best]),
        "n_nonzero_grad": int(improving.size),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--json", default="data/tables/gradient-probe-n16-lam2.0.json", help="output JSON path"
    )
    args = p.parse_args()
    rows = []
    for name, (label, paulis, reference) in CONFIGS.items():
        row = gradient_probe(paulis, reference, N, H, LAM, EPS)
        row.update(config=name, label=label)
        rows.append(row)
        print(
            f"{label:28s} E_ref={row['e_ref']:.6f}  max|grad|={row['max_abs_grad']:.6e}  "
            f"nonzero_grad={row['n_nonzero_grad']:d}/{row['n_generators']:d}  "
            f"best_gen={row['best_generator']}  best_grad={row['best_grad']:.6e}"
        )
    with open(args.json, "w") as fh:
        json.dump({"n_qubits": N, "h": H, "lam": LAM, "eps": EPS, "tol": TOL, "rows": rows}, fh, indent=2)


if __name__ == "__main__":
    main()
