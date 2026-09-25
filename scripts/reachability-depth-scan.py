"""Sequence-length (L) reachability scan across the three canonical token pools.

Quantifies the *finite-sequence-length* term, one
of three candidate causes of the observed collective-pool accuracy floors (the
others being discrete token angles and search/training effects). For the
collective pool there is no group-theoretic obstruction: the LMG ground state
is entirely even-parity, |J,J> is even-parity, and su(J+1) is transitive on the
even sublattice's unit sphere, so the pool's Lie group reaches the ground
state exactly at every (N, lambda). The even sublattice has dimension
J+1 = N/2+1, so specifying a state in it up to norm and global phase costs
N real parameters; each pool token carries exactly one refinable angle, so the
counting bound is L >= N.

This script measures where that bound (and its pairwise-pool analogues) bite,
using RANDOM generator sequences drawn from the real production pools
(`spingqe_lmg.pools.build_pool`) and, by default, RANDOM initial angles
(`--init-angles token` starts from each token's discrete pool angle instead,
reproducing the production refinement starting point). No training of any
kind. Any depth at which random sequences reach machine precision is a depth
at which the generative sequence model cannot be contributing.

Fixed across all three pools: gamma = 0, h = 1, init_state = "zero" (as in
the L=12 pool-accuracy grid). Collective refinement runs in the (N+1)-dim Dicke sector
(`spingqe_lmg.evaluators.dicke.dicke_refine_tokens`, polynomial cost); the two
pairwise pools run full 2^N statevector refinement
(`spingqe_lmg.refine.refine_angles`) on a lightning device chosen by N
(`lightning.qubit` below N=16, `lightning.gpu` at and above).

Streams one CSV row per (pool, N, lambda, L, restart) to stdout-adjacent file
as it completes, so a killed job yields a usable partial curve.

Run on a login node for collective only (CPU, dimensions are N/2+1; minutes).
Pairwise pools need a GPU node (full statevector simulation):
    python scripts/reachability-depth-scan.py --pool collective --out out.csv
    srun ... python scripts/reachability-depth-scan.py --pool pairwise-ext-mf --out out.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from spingqe_lmg.config import HamiltonianConfig, PoolConfig
from spingqe_lmg.enums import Connectivity, HamiltonianKind, PoolKind
from spingqe_lmg.evaluators.dicke import DickeSpace, dicke_refine_tokens
from spingqe_lmg.exact import ground_energy
from spingqe_lmg.hamiltonians import lmg_hamiltonian
from spingqe_lmg.pools import build_pool
from spingqe_lmg.refine import refine_angles

H_FIELD = 1.0
GAMMA = 0.0
LIGHTNING_GPU_THRESHOLD = 16  # lightning.qubit below this N, lightning.gpu at/above
REACHED_TOL = 1e-10  # L-BFGS convergence floor

# PoolConfig fields matching the three pool variants of the L=12 pool-accuracy
# grid exactly. No angle_scale override: that grid uses the unscaled
# DEFAULT_ANGLES.
POOL_CONFIGS = {
    "collective": PoolConfig(kind=PoolKind.COLLECTIVE),
    "pairwise-all": PoolConfig(
        kind=PoolKind.PAULI_PAIR, connectivity=Connectivity.ALL, paulis=("ZZ", "XX", "YY")
    ),
    "pairwise-ext-mf": PoolConfig(
        kind=PoolKind.PAULI_PAIR,
        connectivity=Connectivity.ALL,
        paulis=("ZZ", "XX", "YY", "YZ", "XY"),
    ),
}

CSV_FIELDS = (
    "pool",
    "n_qubits",
    "lam",
    "seq_len",
    "restart",
    "rel_error",
    "n_evaluations",
    "seconds",
    "init_angles",
    "device",
    "reached",
)


def _device_for(n_qubits: int) -> str:
    return "lightning.gpu" if n_qubits >= LIGHTNING_GPU_THRESHOLD else "lightning.qubit"


def _run_one(
    pool_name: str,
    pool_ops,
    n_qubits: int,
    lam: float,
    seq_len: int,
    rng: np.random.Generator,
    init_angles: str,
    dicke_space: DickeSpace | None,
    ham,
) -> dict:
    seq = rng.integers(0, len(pool_ops), size=seq_len)
    x0 = rng.uniform(-np.pi, np.pi, size=seq_len) if init_angles == "random" else None

    t0 = time.perf_counter()
    if pool_name == "collective":
        tokens = [0] + [int(i) + 1 for i in seq]
        result = dicke_refine_tokens(tokens, pool_ops, dicke_space, x0=x0)
        device = "dicke"
        n_evaluations = result.n_evaluations
    else:
        ops = [pool_ops[i] for i in seq]
        device = _device_for(n_qubits)
        result = refine_angles(ops, ham, n_qubits, device_name=device, x0=x0)
        n_evaluations = result.n_evaluations
    seconds = time.perf_counter() - t0

    return {
        "energy": result.energy,
        "n_evaluations": n_evaluations,
        "seconds": seconds,
        "device": device,
    }


def scan(
    pools: list[str],
    ns: list[int],
    lambdas: list[float],
    seq_lens: list[int],
    restarts: int,
    init_angles: str,
    seed: int,
    out_path: Path,
) -> None:
    rng = np.random.default_rng(seed)
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for pool_name in pools:
            pool_cfg = POOL_CONFIGS[pool_name]
            for n_qubits in ns:
                pool_ops = build_pool(pool_cfg, n_qubits)
                for lam in lambdas:
                    hc = HamiltonianConfig(
                        kind=HamiltonianKind.LMG, n_qubits=n_qubits, h=H_FIELD, lam=lam, gamma=GAMMA
                    )
                    e_exact = ground_energy(hc)
                    dicke_space = (
                        DickeSpace.build(n_qubits, H_FIELD, lam, GAMMA)
                        if pool_name == "collective"
                        else None
                    )
                    ham = (
                        lmg_hamiltonian(n_qubits, h=H_FIELD, lam=lam, gamma=GAMMA)
                        if pool_name != "collective"
                        else None
                    )
                    for seq_len in seq_lens:
                        for restart in range(restarts):
                            r = _run_one(
                                pool_name,
                                pool_ops,
                                n_qubits,
                                lam,
                                seq_len,
                                rng,
                                init_angles,
                                dicke_space,
                                ham,
                            )
                            rel_error = abs((r["energy"] - e_exact) / e_exact)
                            writer.writerow(
                                {
                                    "pool": pool_name,
                                    "n_qubits": n_qubits,
                                    "lam": lam,
                                    "seq_len": seq_len,
                                    "restart": restart,
                                    "rel_error": rel_error,
                                    "n_evaluations": r["n_evaluations"],
                                    "seconds": r["seconds"],
                                    "init_angles": init_angles,
                                    "device": r["device"],
                                    "reached": rel_error < REACHED_TOL,
                                }
                            )
                            f.flush()
                            print(
                                f"{pool_name:16s} N={n_qubits:3d} lam={lam:4.1f} "
                                f"L={seq_len:3d} restart={restart:2d} "
                                f"rel_error={rel_error:.3e} ({r['seconds']:.2f}s)",
                                file=sys.stderr,
                            )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--pool",
        action="append",
        choices=sorted(POOL_CONFIGS),
        dest="pools",
        help="repeatable; default: all three",
    )
    p.add_argument("--n-qubits", type=int, nargs="+", default=[8, 10, 12, 14, 16, 20])
    p.add_argument("--lam", type=float, nargs="+", default=[1.0, 2.0])
    p.add_argument("--seq-len", type=int, nargs="+", default=[8, 12, 16, 20, 26, 32])
    p.add_argument("--restarts", type=int, default=12)
    p.add_argument("--init-angles", choices=("random", "token"), default="random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True, help="CSV path; appended to if it exists")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    pools = args.pools or sorted(POOL_CONFIGS)
    scan(
        pools=pools,
        ns=args.n_qubits,
        lambdas=args.lam,
        seq_lens=args.seq_len,
        restarts=args.restarts,
        init_angles=args.init_angles,
        seed=args.seed,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
