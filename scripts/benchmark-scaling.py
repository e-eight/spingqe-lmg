"""Unified GQE benchmark: backends x N, Dicke sector, strong scaling, RSS.

Replaces four scripts: benchmark-sc.py, benchmark-scaling-pinned.py,
benchmark-scaling-statevec.py, benchmark-evaluators.py.  Produces a single CSV
with all timing data consumed by evaluator-crossover and strong-scaling figures
(SC workshop paper).

Modes (--bench):
  backends   pairwise pool across lightning.qubit / lightning.gpu / CUDA-Q v1
             / incremental (CPU) / parity_even (CPU) /
             parity_even_gpu / incremental_gpu at N = 12--28.  Each (N, backend) pair
             runs in a subprocess for isolation (no cross-N compilation tax).
             RSS is collected from the same subprocess.
  dicke      collective pool on the Dicke evaluator at N matching the backends
             sweep plus showcase sizes (50, 100, 500, 1000).  In-process
             (subprocess overhead would dominate µs-scale runs).
  scaling    strong scaling of candidate-sequence energy evaluation across
             n_jobs = 1--64; --evaluator selects the backend, --device-name
             selects the PennyLane device (when --evaluator lightning),
             --pinned records thread-pinning status
  all        run backends, then dicke, then scaling (default)

--backends  comma-separated subset of backends to run (default: all).
            e.g. --backends incremental,parity_even  runs only those two.
            Valid: lightning.qubit, lightning.gpu, cudaq_v1, incremental, parity_even,
                   parity_even_gpu, incremental_gpu

--repeat    number of independent repetitions per cell, each with a distinct
            random seed; produces one CSV row per repetition with a 'rep'
            column (default: 1).  Repetitions run inside a single subprocess
            invocation for --bench backends, or in an outer loop for
            --bench dicke and --bench scaling.

Usage:
  python scripts/benchmark-scaling.py --bench all
  python scripts/benchmark-scaling.py --bench backends --output-csv results.csv
  python scripts/benchmark-scaling.py --bench backends --backends incremental,parity_even
  python scripts/benchmark-scaling.py --bench backends --backends parity_even --n-min 20 --n-max 24
  python scripts/benchmark-scaling.py --bench backends --repeat 5 \\
                                      --backends incremental --n-min 12 --n-max 26
  python scripts/benchmark-scaling.py --bench scaling --evaluator lightning \\
                                    --device-name lightning.gpu
  python scripts/benchmark-scaling.py --bench scaling --evaluator incremental --pinned
"""

from __future__ import annotations

import argparse
import csv
import os
import resource
import subprocess
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="Snapshots are not supported")

from spingqe_lmg.enums import PoolKind  # noqa: E402
from spingqe_lmg.evaluators.dicke import DickeEvaluator  # noqa: E402
from spingqe_lmg.evaluators.pennylane import PennyLaneEvaluator  # noqa: E402
from spingqe_lmg.hamiltonians import lmg_hamiltonian  # noqa: E402
from spingqe_lmg.pools import build_collective_pool, build_pauli_pair_pool  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SEQ_LEN = 12
FULL_N = (12, 14, 16, 18, 20, 22, 24, 26, 28)
DICKE_N = FULL_N + (50, 100, 500, 1000)

# ---------------------------------------------------------------------------
# Backend registry for subprocess dispatch
# ---------------------------------------------------------------------------
_BACKENDS = [
    "lightning.qubit",
    "lightning.gpu",
    "cudaq_v1",
    "incremental",
    "parity_even",
    "parity_even_gpu",
    "incremental_gpu",
    "lightning.qubit (grouped)",
    "lightning.gpu (grouped)",
]

# ---------------------------------------------------------------------------
# CSV row accumulator
# ---------------------------------------------------------------------------
_rows: list[dict] = []
_flushed_count = 0
_output_path: str | None = None


def _add_row(
    benchmark: str,
    n_qubits: int,
    backend: str,
    pool_kind: str,
    n_jobs: int = 1,
    n_sequences: int = 0,
    seq_len: int = SEQ_LEN,
    s_per_seq: float | None = None,
    wall_clock_s: float | None = None,
    pinned: bool = False,
    target: str = "",
    peak_rss_gb: float | None = None,
    *,
    rss_col: str = "peak_rss_gb",
    rep: int = 0,
) -> None:
    _rows.append(
        {
            "benchmark": benchmark,
            "n_qubits": n_qubits,
            "backend": backend,
            "pool_kind": pool_kind,
            "n_jobs": n_jobs,
            "n_sequences": n_sequences,
            "seq_len": seq_len,
            "s_per_seq": f"{s_per_seq:.6f}" if s_per_seq is not None else "",
            "wall_clock_s": f"{wall_clock_s:.2f}" if wall_clock_s is not None else "",
            rss_col: f"{peak_rss_gb:.2f}" if peak_rss_gb is not None else "",
            "pinned": str(pinned),
            "target": target,
            "rep": str(rep),
        }
    )


def _current_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _write_csv(path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_rows[0]) if _rows else [])
        writer.writeheader()
        writer.writerows(_rows)
    print(f"\nWrote {len(_rows)} rows to {path}")


def _flush_rows(path: str) -> None:
    """Append rows accumulated since last flush to *path*."""
    global _flushed_count
    if not _rows or _flushed_count >= len(_rows):
        return
    new_rows = _rows[_flushed_count:]
    mode = "a" if os.path.exists(path) else "w"
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_rows[0]))
        if mode == "w":
            writer.writeheader()
        writer.writerows(new_rows)
    _flushed_count = len(_rows)


def _write_csv_subset(path: str, rows: list[dict]) -> None:
    """Merge *rows* into *path*, preserving existing rows and deduplicating."""
    if not rows:
        print(f"No new rows to write to {path}, skipping.")
        return
    existing: dict[tuple, dict] = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                existing[_row_key(row)] = row
    for row in rows:
        existing[_row_key(row)] = row
    merged = sorted(existing.values(), key=_row_key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows(merged)
    print(
        f"Merged {len(rows)} new rows + {len(existing) - len(rows)} existing → "
        f"{len(merged)} total rows written to {path}"
    )


def _row_key(row: dict) -> tuple:
    return (
        row["benchmark"],
        int(row["n_qubits"]),
        row["backend"],
        int(row["n_jobs"]),
        int(row["seq_len"]),
        int(row.get("rep", 0)),
    )


def _load_existing_keys(path: str) -> frozenset:
    """Return keys for rows already written to *path*; empty frozenset if absent."""
    if not os.path.exists(path):
        return frozenset()
    with open(path, newline="") as f:
        return frozenset(_row_key(row) for row in csv.DictReader(f))


# ---------------------------------------------------------------------------
# Timing helper (in-process only)
# ---------------------------------------------------------------------------
def _time_evaluator(evaluator, idx_batch: np.ndarray) -> float:
    # Warmup with a sequence NOT in the timed batch so the timed call is always
    # a cache miss. Shifting by n_pool//2 guarantees distinct indices for any
    # pool size >= 2 (collective ~3 ops through pairwise ~10k ops).
    n_pool = len(evaluator.pool)
    warmup = (idx_batch[:1] + n_pool // 2) % n_pool
    evaluator(warmup)
    t0 = time.time()
    evaluator(idx_batch)
    return (time.time() - t0) / len(idx_batch)


# ---------------------------------------------------------------------------
# Subprocess worker: timed energy evaluation for one (N, backend) pair
# ---------------------------------------------------------------------------
def _timing_worker(
    n: int,
    backend: str,
    h: float,
    lam: float,
    gamma: float,
    seq_len: int,
    n_seq: int,
    repeat: int = 1,
) -> None:
    """Subprocess entry: time *repeat* distinct-seed evaluations.

    Prints one ``RESULT:<rep>,<s_per_seq>,<rss_kb>`` line per repetition
    with ``flush=True`` so the driver can recover partial results even if
    this subprocess is killed by the OOM killer mid-run.
    """
    ham = lmg_hamiltonian(n, h=h, lam=lam, gamma=gamma)
    pool = build_pauli_pair_pool(n)

    if backend == "lightning.qubit":
        ev = PennyLaneEvaluator(pool, ham, n, device_name="lightning.qubit")
    elif backend == "lightning.qubit (grouped)":
        from spingqe_lmg.evaluators.pennylane import PennyLaneGroupedEvaluator

        ev = PennyLaneGroupedEvaluator(pool, ham, n, device_name="lightning.qubit")
    elif backend == "lightning.gpu":
        ev = PennyLaneEvaluator(pool, ham, n, device_name="lightning.gpu")
    elif backend == "lightning.gpu (grouped)":
        from spingqe_lmg.evaluators.pennylane import PennyLaneGroupedEvaluator

        ev = PennyLaneGroupedEvaluator(pool, ham, n, device_name="lightning.gpu")
    elif backend == "cudaq_v1":
        from spingqe_lmg.evaluators.cudaq import CudaQEvaluator

        ev = CudaQEvaluator(pool, ham, n, chain_states=False)
    elif backend == "incremental":
        from spingqe_lmg.evaluators.statevec import IncrementalEvaluator

        ev = IncrementalEvaluator(pool, ham, n)
    elif backend == "parity_even":
        from spingqe_lmg.evaluators.statevec import ParityEvenEvaluator

        ev = ParityEvenEvaluator(pool, ham, n)
    elif backend == "parity_even_gpu":
        from spingqe_lmg.evaluators.statevec_gpu import (
            ParityEvenGPUEvaluator,
        )

        ev = ParityEvenGPUEvaluator(pool, ham, n)
    elif backend == "incremental_gpu":
        from spingqe_lmg.evaluators.statevec_gpu import IncrementalGPUEvaluator

        ev = IncrementalGPUEvaluator(pool, ham, n)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # --- timed repetitions, each with a distinct seed ---
    # _time_evaluator handles its own warmup internally (shifted index),
    # so each rep gets a clean post-JIT measurement without cross-rep
    # cache pollution.  Results are printed immediately so the driver
    # can recover partial data if this subprocess is killed mid-N.
    for rep in range(repeat):
        rng = np.random.default_rng(7 + rep)
        idx = rng.integers(0, len(pool), size=(n_seq, seq_len))
        s_per_seq = _time_evaluator(ev, idx)
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"RESULT:{rep},{s_per_seq:.6f},{rss_kb}", flush=True)


# ---------------------------------------------------------------------------
# Benchmark: all backends on pairwise pool (subprocess-isolated per data point)
# ---------------------------------------------------------------------------
def _bench_backends(args: argparse.Namespace, skip_keys: frozenset = frozenset()) -> None:
    n_seq = args.n_seq or 20
    seq_len = args.seq_len
    backend_filter = _parse_backend_filter(args.backends)

    backends_to_run = _BACKENDS
    if backend_filter is not None:
        backends_to_run = [b for b in _BACKENDS if b in backend_filter]

    for n in args.n_range:
        print(f"N={n} pairwise (pool ~{1 + 3 * n + n * (n - 1) // 2} ops, {seq_len}-gate):")
        for backend in backends_to_run:
            # Resume: if rep=0 is already present for this cell, skip
            # (all reps are written atomically by one subprocess invocation).
            if ("backend", n, backend, 1, seq_len, 0) in skip_keys:
                print(f"  {backend:22s} skipped (resume)")
                continue
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--timing-worker",
                        str(n),
                        backend,
                        str(args.h),
                        str(args.lam),
                        str(args.gamma),
                        str(seq_len),
                        str(n_seq),
                        str(args.repeat),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=int(600 * (2 ** max(n - 20, 0))),
                )
                rep_rows = 0
                for line in result.stdout.strip().splitlines():
                    if line.startswith("RESULT:"):
                        parts = line[len("RESULT:"):].strip().split(",")
                        rep_i = int(parts[0])
                        s_per_seq = float(parts[1])
                        rss_gb = int(parts[2]) / 1e6
                        _add_row(
                            "backend",
                            n,
                            backend,
                            PoolKind.PAULI_PAIR.value,
                            n_sequences=n_seq,
                            seq_len=seq_len,
                            s_per_seq=s_per_seq,
                            peak_rss_gb=rss_gb,
                            rss_col="process_rss_gb",
                            rep=rep_i,
                        )
                        rep_rows += 1
                if result.returncode == 0:
                    print(f"  {backend:22s} {rep_rows} reps recorded")
                elif rep_rows > 0:
                    print(
                        f"  {backend:22s} {rep_rows}/{args.repeat} reps saved "
                        f"(rc={result.returncode}) "
                        f"stderr: {result.stderr[:200]}"
                    )
                else:
                    print(
                        f"  {backend:22s} FAILED (rc={result.returncode}) "
                        f"stderr: {result.stderr[:200]}"
                    )
            except subprocess.TimeoutExpired:
                print(f"  {backend:22s} TIMEOUT")
            except Exception as exc:  # noqa: BLE001
                print(f"  {backend:22s} unavailable ({type(exc).__name__})")
        if args.output_csv:
            _flush_rows(args.output_csv)
        print(flush=True)


# ---------------------------------------------------------------------------
# Benchmark: Dicke-sector evaluator on collective pool (in-process)
# ---------------------------------------------------------------------------
def _bench_dicke(args: argparse.Namespace, skip_keys: frozenset = frozenset()) -> None:
    n_seq = args.n_seq or 20
    seq_len = args.seq_len
    extra = tuple(n for n in (50, 100, 500, 1000) if n not in set(args.n_range))
    dicke_n = args.n_range + extra

    for rep in range(args.repeat):
        repo_label = f"  [rep {rep}]" if args.repeat > 1 else ""
        print(f"Dicke backend (collective pool, (N+1)-dim sector):{repo_label}")
        rng = np.random.default_rng(11 + rep)
        for n in dicke_n:
            if ("dicke", n, "dicke", 1, seq_len, rep) in skip_keys:
                print(f"  N={n:5d}  skipped (resume)", flush=True)
                continue
            pool = build_collective_pool(n)
            idx = rng.integers(0, len(pool), size=(n_seq, seq_len))
            ev = DickeEvaluator(pool, n, args.h, args.lam, args.gamma)
            val = _time_evaluator(ev, idx)
            rss = _current_rss_gb()
            _add_row(
                "dicke",
                n,
                "dicke",
                PoolKind.COLLECTIVE.value,
                n_sequences=n_seq,
                seq_len=seq_len,
                s_per_seq=val,
                peak_rss_gb=rss,
                rss_col="process_rss_gb",
                rep=rep,
            )
            if args.output_csv:
                _flush_rows(args.output_csv)
            print(f"  N={n:5d}  {val:9.5f} s/seq", flush=True)
        print(flush=True)


# ---------------------------------------------------------------------------
# Benchmark: strong scaling (n_jobs sweep, in-process)
# ---------------------------------------------------------------------------
def _bench_scaling(args: argparse.Namespace, skip_keys: frozenset = frozenset()) -> None:
    n = args.scaling_n
    n_seq = args.n_seq or args.scaling_batch
    seq_len = args.seq_len
    pinned = args.pinned
    evaluator_name = args.evaluator
    scaling_jobs = args.scaling_jobs

    if pinned:
        print(
            f"thread env: OMP={os.environ.get('OMP_NUM_THREADS')} "
            f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')} "
            f"MKL={os.environ.get('MKL_NUM_THREADS')}",
            flush=True,
        )

    if evaluator_name == "incremental":
        from spingqe_lmg.evaluators.statevec import IncrementalEvaluator

        ev_cls = IncrementalEvaluator
        backend_label = "incremental"
    elif evaluator_name == "parity_even":
        from spingqe_lmg.evaluators.statevec import ParityEvenEvaluator

        ev_cls = ParityEvenEvaluator
        backend_label = "parity_even"
    else:
        if args.device_name.endswith(" (grouped)"):
            from spingqe_lmg.evaluators.pennylane import PennyLaneGroupedEvaluator

            ev_cls = PennyLaneGroupedEvaluator
            device_name = args.device_name.removesuffix(" (grouped)")
            backend_label = args.device_name
        else:
            ev_cls = PennyLaneEvaluator
            device_name = args.device_name
            backend_label = args.device_name

    for rep in range(args.repeat):
        repo_label = f"  [rep {rep}]" if args.repeat > 1 else ""
        label = f"{backend_label}" + (" (pinned)" if pinned else "") + repo_label
        print(f"Strong scaling: {label}, N={n}, {n_seq} sequences:")

        rng = np.random.default_rng(13 + rep)
        ham = lmg_hamiltonian(n, h=args.h, lam=args.lam, gamma=args.gamma)
        pool = build_pauli_pair_pool(n)
        idx = rng.integers(0, len(pool), size=(n_seq, seq_len))

        base = None
        for j in scaling_jobs:
            if ("scaling", n, backend_label, j, seq_len, rep) in skip_keys:
                print(f"  n_jobs={j:3d}  skipped (resume)", flush=True)
                continue
            if evaluator_name == "incremental":
                ev = ev_cls(pool, ham, n, n_jobs=j)
            elif evaluator_name == "parity_even":
                ev = ev_cls(pool, ham, n, n_jobs=j)
            else:
                ev = ev_cls(pool, ham, n, device_name=device_name, n_jobs=j)
            n_pool = len(pool)
            warmup_idx = (idx + n_pool // 2) % n_pool
            ev(warmup_idx)
            t0 = time.time()
            ev(idx)
            dt = time.time() - t0
            base = base if base is not None else dt
            rss = _current_rss_gb()
            _add_row(
                "scaling",
                n,
                backend_label,
                PoolKind.PAULI_PAIR.value,
                n_jobs=j,
                n_sequences=n_seq,
                seq_len=seq_len,
                wall_clock_s=dt,
                pinned=pinned,
                peak_rss_gb=rss,
                rss_col="process_rss_gb",
                rep=rep,
            )
            print(
                f"  n_jobs={j:3d}  {dt:8.2f} s  ({n_seq / dt:7.1f} seq/s)  "
                f"speedup x{base / dt:5.1f}",
                flush=True,
            )
        print(flush=True)
        if args.output_csv:
            _flush_rows(args.output_csv)


def _parse_backend_filter(raw: str | None) -> set[str] | None:
    """Parse --backends comma-separated string into a set, or None for all.

    ``+`` in a backend name is replaced with a space before matching, so
    ``lightning.qubit+(grouped)`` maps to ``lightning.qubit (grouped)``
    without requiring shell quoting through the BENCH_ARGS pipeline.
    """
    if raw is None:
        return None
    return {b.strip().replace("+", " ") for b in raw.split(",") if b.strip()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    # --timing-worker is an internal flag, handled before argparse
    if argv is None:
        argv = sys.argv[1:]
    if "--timing-worker" in (argv or []):
        idx = argv.index("--timing-worker")
        n = int(argv[idx + 1])
        backend = argv[idx + 2]
        h = float(argv[idx + 3])
        lam = float(argv[idx + 4])
        gamma = float(argv[idx + 5])
        seq_len = int(argv[idx + 6])
        n_seq = int(argv[idx + 7])
        repeat = int(argv[idx + 8]) if len(argv) > idx + 8 else 1
        _timing_worker(n, backend, h, lam, gamma, seq_len, n_seq, repeat=repeat)
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench",
        choices=["backends", "dicke", "scaling", "all"],
        default="all",
        help="which benchmark(s) to run",
    )
    parser.add_argument(
        "--evaluator",
        choices=["lightning", "incremental", "parity_even"],
        default="lightning",
        help="backend for --bench scaling",
    )
    parser.add_argument(
        "--device-name",
        default="lightning.qubit",
        help="PennyLane device name when --evaluator lightning "
        "(e.g. lightning.qubit, lightning.gpu). "
        "Ignored for other evaluators.  Default: lightning.qubit.",
    )
    parser.add_argument(
        "--backends",
        metavar="LIST",
        help="comma-separated subset of backends for --bench backends "
        f"(e.g. --backends {','.join(_BACKENDS[:2])}).  Valid: {','.join(_BACKENDS)}.  "
        "Default: all.",
    )
    parser.add_argument(
        "--pinned",
        action="store_true",
        help="record thread-pinning status (scaling mode)",
    )
    parser.add_argument(
        "--output-csv",
        metavar="PATH",
        help="write structured CSV output",
    )
    parser.add_argument(
        "--n-seq",
        type=int,
        help="number of random sequences (default: 20 for backends/dicke, 128 for scaling)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=SEQ_LEN,
        help=f"gates per sequence (default: {SEQ_LEN})",
    )
    parser.add_argument(
        "--n-min",
        type=int,
        default=min(FULL_N),
        help=f"smallest N to benchmark (default: {min(FULL_N)})",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=max(FULL_N),
        help=f"largest N to benchmark (default: {max(FULL_N)})",
    )
    parser.add_argument(
        "--n-step",
        type=int,
        default=2,
        help="step between N values (default: 2)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="number of independent repetitions per cell (each with a distinct seed); "
        "produces one CSV row per repetition with a 'rep' column (default: 1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip cells already present in --output-csv (requires --output-csv)",
    )
    parser.add_argument(
        "--h",
        type=float,
        default=1.0,
        help="LMG transverse field strength (default: 1.0)",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=1.5,
        help="LMG interaction strength lambda (default: 1.5)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help="LMG anisotropy gamma (default: 0.0)",
    )
    parser.add_argument(
        "--scaling-n",
        type=int,
        default=20,
        help="N for strong-scaling benchmark (default: 20)",
    )
    parser.add_argument(
        "--scaling-batch",
        type=int,
        default=128,
        help="number of sequences for strong-scaling benchmark (default: 128)",
    )
    parser.add_argument(
        "--scaling-jobs",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32, 64],
        help="n_jobs values for strong-scaling sweep (default: 1 4 8 16 32 64)",
    )
    args = parser.parse_args(argv)
    if args.n_min > args.n_max:
        parser.error(f"--n-min ({args.n_min}) must be <= --n-max ({args.n_max})")
    args.n_range = tuple(range(args.n_min, args.n_max + 1, args.n_step))

    skip_keys: frozenset = frozenset()
    if args.resume and args.output_csv:
        skip_keys = _load_existing_keys(args.output_csv)
        if skip_keys:
            print(f"Resuming: skipping {len(skip_keys)} existing cells from {args.output_csv}")

    if args.bench in ("backends", "all"):
        _bench_backends(args, skip_keys=skip_keys)
    if args.bench in ("dicke", "all"):
        _bench_dicke(args, skip_keys=skip_keys)
    if args.bench in ("scaling", "all"):
        _bench_scaling(args, skip_keys=skip_keys)

    rss_gb = _current_rss_gb()
    print(f"peak RSS (driver process): {rss_gb:.2f} GB")

    if args.output_csv:
        _flush_rows(args.output_csv)
        if _flushed_count > 0:
            print(f"\nAll rows flushed to {args.output_csv}")


if __name__ == "__main__":
    main()
