#!/usr/bin/env python
"""Single-token descent test: does any one pool token improve on the reference state?

Applies every distinct token in a pool once, at its discrete angle, to that
configuration's own reference state, and scores the resulting energy against the
reference energy. A configuration no single token improves is a token-level local
minimum. The test needs relative energies only, so it runs without a known ground
state -- this is the cheap probe of the capability check (paper Sec. V,
Table III).

Usage:
    python scripts/descent-test.py                    # all configs, N=16, lambda=2.0
    python scripts/descent-test.py --json data/tables/descent-test-n16-lam2.0.json
    python scripts/descent-test.py --config collective --json out.json
"""

import argparse
import json

import numpy as np

from spingqe_lmg.evaluators.statevec import IncrementalEvaluator
from spingqe_lmg.hamiltonians import lmg_hamiltonian, lmg_mean_field_angle
from spingqe_lmg.pools import build_collective_pool, build_pauli_pair_pool

STD_PAULIS = ("ZZ", "XX", "YY")
EXT_PAULIS = ("ZZ", "XX", "YY", "YZ", "XY")

# (label, pool factory, reference state) -- reference is "zero" or "mean_field".
CONFIGS = {
    "std-pairwise-zero": ("std. pairwise, zero", STD_PAULIS, "zero"),
    "std-pairwise-mf": ("std. pairwise, mean field", STD_PAULIS, "mean_field"),
    "ext-pairwise-zero": ("ext. pairwise, zero", EXT_PAULIS, "zero"),
    "ext-pairwise-mf": ("ext. pairwise, mean field", EXT_PAULIS, "mean_field"),
    "collective-zero": ("collective, zero", None, "zero"),
}


def build_pool(paulis, n_qubits):
    """Collective pool when paulis is None, otherwise the pairwise pool."""
    if paulis is None:
        return build_collective_pool(n_qubits)
    return build_pauli_pair_pool(n_qubits, paulis=paulis)


def descent_test(paulis, reference, n_qubits, h, lam, tol):
    """Energies of every single-token circuit against the reference energy."""
    pool = build_pool(paulis, n_qubits)
    ham = lmg_hamiltonian(n_qubits, h=h, lam=lam)
    init_angle = lmg_mean_field_angle(h, lam) if reference == "mean_field" else 0.0
    ev = IncrementalEvaluator(pool, ham, n_qubits, init_angle=init_angle)

    e_ref = ev.initial_energy()
    # One sequence per token; the evaluator returns per-prefix energies, and each
    # sequence has a single prefix.
    energies = ev(np.arange(len(pool), dtype=np.int64).reshape(-1, 1))[:, 0]

    best = int(np.argmin(energies))
    improving = np.flatnonzero(energies < e_ref - tol)
    return {
        "reference": reference,
        "mean_field_angle": float(init_angle),
        "n_tokens": len(pool),
        "e_ref": float(e_ref),
        "e_best": float(energies[best]),
        "best_token": pool[best].label,
        "gain": float(e_ref - energies[best]),
        "n_improving": int(improving.size),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=16, help="number of spins/qubits")
    p.add_argument("--h", type=float, default=1.0, help="LMG field strength (energy scale)")
    p.add_argument("--lam", type=float, default=2.0, help="LMG coupling")
    p.add_argument(
        "--config",
        action="append",
        choices=sorted(CONFIGS),
        help="configuration to test (repeatable; default: all)",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        help="a token counts as improving when it lowers the energy by more than this",
    )
    p.add_argument("--json", help="write results to this path as JSON")
    args = p.parse_args()

    names = args.config or sorted(CONFIGS)
    rows = []
    for name in names:
        label, paulis, reference = CONFIGS[name]
        row = descent_test(paulis, reference, args.n, args.h, args.lam, args.tol)
        row.update(config=name, label=label)
        rows.append(row)
        print(
            f"{label:28s} E_ref={row['e_ref']:.6f}  best={row['e_best']:.6f}  "
            f"improving={row['n_improving']:d}/{row['n_tokens']:d}  "
            f"gain={row['gain']:.6f}  best_token={row['best_token']}"
        )

    if args.json:
        payload = {"n_qubits": args.n, "h": args.h, "lam": args.lam, "tol": args.tol, "rows": rows}
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
