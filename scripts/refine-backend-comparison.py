"""Is angle refinement backend-neutral, and which lightning backend is faster at each N?

Underwrites the manuscript's refinement-backend policy: pick whichever lightning
backend is cheaper at a given N, since the refined energy does not depend on the
choice.  Unlike the *training* backend, whose per-step rounding compounds
chaotically over 700 epochs (2026-07-25 depth analysis, section 3), refinement is
a deterministic L-BFGS-B solve from a fixed sequence and fixed initial angles, so
backend differences should stay at float64 noise.  This script measures that
rather than assuming it, and times each backend so the N-dependent choice has
data behind it.

Only the pairwise pools are covered: DickeEvaluator.refine_tokens refines inside
the (N+1)-dimensional sector and ignores qml_device entirely, so the collective
column's refinement is backend-independent by construction.

Usage:
    python scripts/refine-backend-comparison.py --n-qubits 12,16,20
"""

import argparse
import json
import time

import numpy as np

from spingqe_lmg.config import HamiltonianConfig, PoolConfig
from spingqe_lmg.enums import Connectivity, HamiltonianKind, PoolKind, RefineGradient
from spingqe_lmg.hamiltonians import build_hamiltonian
from spingqe_lmg.pools import build_pool
from spingqe_lmg.refine import refine_tokens

# Extended pairwise (the parity-odd pool that forecloses the cheap evaluators and
# therefore always refines through PennyLane).
EXT_PAULIS = ("ZZ", "XX", "YY", "YZ", "XY")
DEVICES = ["default.qubit", "lightning.qubit", "lightning.gpu"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-qubits", default="12,16,20")
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--seq-len", type=int, default=12)
    ap.add_argument("--n-sequences", type=int, default=3, help="distinct fixed sequences per N")
    ap.add_argument("--devices", default=",".join(DEVICES))
    args = ap.parse_args()

    devices = args.devices.split(",")
    results = []

    for n_qubits in (int(x) for x in args.n_qubits.split(",")):
        pool_cfg = PoolConfig(
            kind=PoolKind.PAULI_PAIR,
            connectivity=Connectivity.ALL,
            paulis=EXT_PAULIS,
        )
        pool = build_pool(pool_cfg, n_qubits)
        ham_cfg = HamiltonianConfig(
            kind=HamiltonianKind.LMG, n_qubits=n_qubits, h=1.0, lam=args.lam, gamma=0.0
        )
        ham = build_hamiltonian(ham_cfg)
        rng = np.random.default_rng(n_qubits * 1000 + int(args.lam * 10))

        print(f"\n=== N={n_qubits}, lam={args.lam}, pool={len(pool)} tokens ===")
        for s in range(args.n_sequences):
            idx = rng.integers(0, len(pool), size=args.seq_len, dtype=np.int64)
            tokens = [0] + [int(i) + 1 for i in idx]
            per_device = {}
            for dev in devices:
                try:
                    t0 = time.perf_counter()
                    res = refine_tokens(
                        tokens,
                        pool,
                        ham,
                        n_qubits,
                        device_name=dev,
                        init_angle=0.0,
                        shots=0,
                        gradient=RefineGradient.ADJOINT,
                    )
                    per_device[dev] = {
                        "energy": float(res.energy),
                        "seconds": time.perf_counter() - t0,
                        "n_evaluations": int(res.n_evaluations),
                    }
                except Exception as exc:  # backend unavailable at this size
                    per_device[dev] = {"error": f"{type(exc).__name__}: {exc}"}

            ok = {d: v for d, v in per_device.items() if "energy" in v}
            energies = [v["energy"] for v in ok.values()]
            spread = max(energies) - min(energies) if len(energies) > 1 else 0.0
            ref = min(abs(e) for e in energies) if energies else 1.0
            print(f"  sequence {s}: energy spread across backends = {spread:.3e} "
                  f"(relative {spread / max(ref, 1e-300):.3e})")
            for d, v in per_device.items():
                if "energy" in v:
                    print(f"    {d:18s} E={v['energy']:.12f}  {v['seconds']:7.2f}s  "
                          f"{v['n_evaluations']} evals")
                else:
                    print(f"    {d:18s} {v['error']}")
            results.append(
                {
                    "n_qubits": n_qubits,
                    "lam": args.lam,
                    "sequence": s,
                    "tokens": tokens,
                    "energy_spread": spread,
                    "relative_spread": spread / max(ref, 1e-300),
                    "per_device": per_device,
                }
            )

    print("\n=== summary: refinement backend agreement ===")
    worst = max((r["relative_spread"] for r in results), default=0.0)
    print(f"worst relative energy spread across backends: {worst:.3e}")
    print("\n=== summary: median refinement wall-time by (N, backend) ===")
    for n_qubits in sorted({r["n_qubits"] for r in results}):
        rows = [r for r in results if r["n_qubits"] == n_qubits]
        parts = []
        for dev in devices:
            secs = [r["per_device"][dev]["seconds"] for r in rows if "seconds" in r["per_device"][dev]]
            parts.append(f"{dev}={np.median(secs):.2f}s" if secs else f"{dev}=n/a")
        print(f"  N={n_qubits}: " + "  ".join(parts))

    print("\n" + json.dumps({"worst_relative_spread": worst, "results": results}, indent=2)[:0])
    out = "refine-backend-comparison.json"
    with open(out, "w") as fh:
        json.dump({"worst_relative_spread": worst, "results": results}, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
