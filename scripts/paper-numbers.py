#!/usr/bin/env python
"""Recompute the in-text numbers of the SC paper from the shipped data.

Every table and figure has its own generator (see README.md); this script
covers the numbers quoted in the prose and footnotes that no table prints.
It reads only ``data/`` and needs no GPU. Each block prints the paper's
claim next to the value recomputed from the data.

Usage::

    python scripts/paper-numbers.py
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "data" / "runs"
TABLES = REPO / "data" / "tables"
FLOOR = 1e-10


def load_runs(campaign: str, sub: str = "") -> list[dict]:
    """One dict per run directory: config, metadata and refined relative error."""
    out = []
    root = RUNS / campaign / sub if sub else RUNS / campaign
    for d in sorted(root.iterdir()):
        if not (d / "metadata.json").exists() or not (d / "config.json").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        meta = json.loads((d / "metadata.json").read_text())
        e0 = meta["ground_energy"]
        out.append(
            {
                "name": d.name,
                "n": cfg["hamiltonian"]["n_qubits"],
                "lam": cfg["hamiltonian"]["lam"],
                "seed": cfg["train"]["seed"],
                "seq_len": cfg["train"]["seq_len"],
                "pool": cfg["pool"]["kind"],
                "init_state": cfg["train"].get("init_state", "zero"),
                "angle_scale": cfg["pool"].get("angle_scale", 0),
                "e_ref": meta["initial_energy"],
                "e_refined": meta["refined_energy"],
                "rel": abs(meta["refined_energy"] - e0) / abs(e0),
            }
        )
    return out


def header(title: str) -> None:
    print(f"\n== {title}")


def onset_table() -> None:
    header("tab:onset -- onset = first L with success fraction >= 0.5 (collective, zero ref)")
    frames = [
        pd.read_csv(TABLES / "lreach-2026-07-26.csv", comment="#"),
        pd.read_csv(TABLES / "lreach-n28-2026-07-28.csv", comment="#"),
    ]
    df = pd.concat(frames)
    col = df[df["pool"] == "collective"]
    print(f"{'N':>3} {'bound':>5} {'lam=1.5':>8} {'lam=2.0':>8}")
    for n in sorted(col["n_qubits"].unique()):
        cells = []
        for lam in (1.5, 2.0):
            sub = col[(col["n_qubits"] == n) & (col["lam"] == lam)].sort_values("seq_len")
            hit = sub[sub["success_fraction"] >= 0.5]
            cells.append(str(int(hit["seq_len"].iloc[0])) if len(hit) else "---")
        print(f"{n:>3} {n:>5} {cells[0]:>8} {cells[1]:>8}")
    n28 = col[col["n_qubits"] == 28].set_index("seq_len")["success_fraction"]
    print(
        f"N=28 success fractions: L=68 {n28[68] * 12:.0f}/12, L=72 {n28[72] * 12:.0f}/12 "
        "(paper: 4/12, 6/12)"
    )

    pw = df[(df["pool"] != "collective") & (df["seq_len"] <= 64)]
    print(
        f"pairwise cells with L<=64 reaching the floor: "
        f"{int((pw['success_fraction'] > 0).sum())} of {len(pw)} (paper: none)"
    )


def onset_scan_cost() -> None:
    header("onset-scan cost -- serial refinement time for one (pool, N, lambda) sweep")
    secs = defaultdict(float)
    for f in (RUNS / "reachability-depth-grid-2026-07-26" / "cells").glob("cell-*.csv"):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                secs[(r["pool"], int(r["n_qubits"]), float(r["lam"]))] += float(r["seconds"])
    col = {k: v for k, v in secs.items() if k[0] == "collective"}
    worst = max(col, key=col.get)
    print(
        f"collective, max over N<=24: {col[worst] / 60:.2f} min at N={worst[1]}, "
        f"lambda={worst[2]} (paper: under six minutes through N=24)"
    )
    pw = {k: v for k, v in secs.items() if k[0] != "collective"}
    worst = max(pw, key=pw.get)
    print(f"pairwise, max: {pw[worst] / 60:.2f} min ({worst[0]}, N={worst[1]}, lambda={worst[2]})")


def floor_length() -> None:
    header("floor length -- seeds at the floor at L=26 (collective, zero ref, 6 seeds)")
    runs = load_runs("collective-dicke-l26")
    cells = defaultdict(list)
    for r in runs:
        cells[(r["n"], r["lam"])].append(r["rel"])
    for (n, lam), rels in sorted(cells.items()):
        k = sum(x <= FLOOR for x in rels)
        print(f"N={n:>2} lambda={lam}: {k}/{len(rels)} at floor, worst {max(rels):.2e}")
    print("(paper: 6/6 for N<=18 at both lambdas; N=20: 3/6 at lambda=1.0, 4/6 at lambda=2.0)")


def n28_arms() -> None:
    header("pre-registered N=28 arms (lambda=2.0, 6 seeds each)")
    runs = load_runs("prospective-depth-n28-2026-07-28", "train")
    by_l = defaultdict(list)
    for r in runs:
        by_l[r["seq_len"]].append(r["rel"])
    for L, rels in sorted(by_l.items()):
        k = sum(x <= FLOOR for x in rels)
        print(f"L={L}: {k}/{len(rels)} at floor, range {min(rels):.2e} .. {max(rels):.2e}")
    ratio = statistics.median(by_l[68]) / statistics.median(by_l[72])
    spread = max(by_l[68]) / min(by_l[68])
    print(f"median(L=68)/median(L=72) = {ratio:.2f} (paper: 1.41)")
    print(f"L=68 max/min = {spread:.2f} (paper: 2.25)")


def pairwise_grid() -> None:
    header("L=12 grid footnote -- pairwise ten-seed spread")
    df = pd.read_csv(TABLES / "pool-accuracy-l12-s1-10-2026-07-29.csv", comment="#")
    pw = df[df["pool"] != "collective"]
    g = pw.groupby(["pool", "N", "lambda"])["refined_rel_error"]
    decades = np.log10(g.max() / g.min())
    print(
        f"pairwise cells: {len(decades)} (paper: 34); max ten-seed range "
        f"{decades.max():.3f} decades (paper: under 0.06); total cells "
        f"{df.groupby(['pool', 'N', 'lambda']).ngroups} (paper: 51)"
    )


def pairwise_vs_reference() -> None:
    header("pairwise runs against their own reference-state energy")
    runs = []
    for camp in (
        "pool-accuracy-l12-s1-10-2026-07-28",
        "pool-accuracy-pairwise-l12-s1-10-2026-07-29",
        "pool-vs-reference-2026-07-26",
    ):
        runs += [r for r in load_runs(camp) if r["pool"] == "pauli_pair"]
    dev = [abs(r["e_ref"] - r["e_refined"]) / abs(r["e_ref"]) for r in runs]
    print(
        f"{len(runs)} pairwise runs; max |E_ref - E_refined| / |E_ref| = "
        f"{max(dev) * 100:.2f}% (paper: within 1.4%)"
    )

    df = pd.read_csv(TABLES / "2x2-table-2026-07-29.csv", comment="#")
    low = df[df["block"] == "lower"]
    row = low[(low["pool"].str.contains("std")) & (low["reference"] == "mean field")]
    for _, r in row.iterrows():
        print(
            f"std. pair., m.f., lambda={r['lam']}: median refined/reference error "
            f"ratio {r['refined_rel_error'] / r['reference_rel_error']:.12f}"
        )
    std_mf = [
        r for r in load_runs("pool-vs-reference-2026-07-26")
        if r["init_state"] == "mean_field"
    ]
    std_mf += [
        r for r in load_runs("pool-accuracy-pairwise-l12-s1-10-2026-07-29")
        if r["init_state"] == "mean_field" and r["n"] == 16 and "ext" not in r["name"]
    ]
    for lam in (1.5, 2.0):
        d = [abs(r["e_refined"] - r["e_ref"]) for r in std_mf if r["lam"] == lam
             and "pairwise-all" in r["name"]]
        if d:
            print(
                f"  per-seed |E_refined - E_ref| at lambda={lam}: "
                f"{sum(x <= 1e-10 for x in d)}/{len(d)} within 1e-10, max {max(d):.1e}"
            )
    print("(paper: refined energy equals the reference energy to twelve digits)")


def two_by_two() -> None:
    header("tab:2x2 prose -- reference-state and pool levers at N=16, L=12")
    df = pd.read_csv(TABLES / "2x2-table-2026-07-29.csv", comment="#")
    up = df[df["block"] == "upper"].set_index(["pool", "reference", "lam"])["refined_rel_error"]
    for lam in (1.5, 2.0):
        for pool in ("std.\\ pair.", "ext.\\ pair."):
            ratio = up[(pool, "zero", lam)] / up[(pool, "mean field", lam)]
            print(f"lambda={lam} {pool}: zero/m.f. = {ratio:.2f}x")
        pools = [up[(p, ref, lam)] for p in ("std.\\ pair.", "ext.\\ pair.")
                 for ref in ("zero",)]
        pools_mf = [up[(p, "mean field", lam)] for p in ("std.\\ pair.", "ext.\\ pair.")]
        gap = max(abs(pools[0] / pools[1] - 1), abs(pools_mf[0] / pools_mf[1] - 1))
        print(f"lambda={lam}: pools at fixed reference differ by at most {gap * 100:.2f}%")
    print("(paper: roughly 6.5x at lambda=1.5, 70x at lambda=2.0; pools within 3%)")
    low = df[df["block"] == "lower"]
    coll = low[low["pool"].str.contains("coll") & (low["reference"] == "zero")]
    for _, r in coll.iterrows():
        print(f"collective, zero ref, lambda={r['lam']}: gain over reference {r['gain']:.2f}x")
    print("(paper: 7.65x at lambda=2.0, roughly 40-fold at lambda=1.5)")
    zero = statistics.median(
        r["rel"] for r in load_runs("pool-accuracy-l12-s1-10-2026-07-28")
        if r["pool"] == "collective" and r["n"] == 16 and r["lam"] == 2.0
    )
    mf = statistics.median(
        r["rel"] for r in load_runs("collective-mf-n16-2026-07-30") if r["lam"] == 2.0
    )
    print(
        f"collective lambda=2.0: zero {zero:.2e}, m.f. {mf:.2e}, ratio {zero / mf:.1f}x "
        "(paper: 2.2e-2, 1.4e-3, 16x)"
    )


def sampler_control() -> None:
    header("uniform random-sampler control (paired against the 3-seed runs)")
    df = pd.read_csv(RUNS / "random-sampler-control-2026-07-25" / "summary.csv")
    df["ratio"] = df["random_refined_rel_error"] / df["gqe_refined_rel_error"]
    print(f"pairs: {len(df)} over pools {sorted(df['pool'].unique())} (paper: 153)")
    pw = df[df["pool"] != "collective"]
    inside = ((pw["ratio"] >= 1 / 1.1) & (pw["ratio"] <= 1.1)).sum()
    print(
        f"pairwise pairs within a factor 1.1 either way: {inside} of {len(pw)} "
        "(paper: 100 of 102)"
    )
    for _, r in pw[(pw["ratio"] < 1 / 1.1) | (pw["ratio"] > 1.1)].iterrows():
        print(f"  outside: {r['pool']} N={r['N']} lambda={r['lambda']} draw {r['draw']}: "
              f"{r['ratio']:.3f}")
    lr = np.log10(df[df["pool"] == "collective"]["ratio"])
    print(
        f"collective log10(random/GQE): median {lr.median():.2f}, sample std "
        f"{lr.std(ddof=1):.2f} decades (paper: median zero, spread 2.2 decades)"
    )


def showcase() -> None:
    header("Fig. dicke-showcase and the N=1000 endpoint (lambda=1.5, L=12, 8/N vocabulary)")
    mf = [r for r in load_runs("dicke-showcase-mf-seeds-2026-08-06") if r["angle_scale"] == 8]
    zero = [r for r in load_runs("dicke-showcase-zero-seeds-2026-08-06") if r["angle_scale"] == 8]

    def med(runs, n):
        return statistics.median(r["rel"] for r in runs if r["n"] == n)

    best = min(r["rel"] for r in mf if r["n"] == 1000)
    print(f"m.f. N=1000: median {med(mf, 1000):.2e} (paper 2.1e-6), best {best:.2e} (paper 1.1e-7)")
    for n in (50, 1000):
        print(
            f"zero/m.f. median at N={n}: {math.log10(med(zero, n) / med(mf, n)):.1f} decades "
            f"(paper: {'one' if n == 50 else 'four'})"
        )
    t = pd.read_csv(TABLES / "n1000-mf-timing-sacct.csv", comment="#")
    mins = [sum(int(x) * 60**i for i, x in enumerate(reversed(e.split(":")))) / 60
            for e in t["elapsed"]]
    print(f"N=1000 job wall time: {min(mins):.1f}-{max(mins):.1f} min, median "
          f"{statistics.median(mins):.1f} (paper: about 6 min)")
    bench = pd.read_csv(TABLES / "bench-merged-2026-07-08.csv")
    s = bench[(bench["n_qubits"] == 1000)]["s_per_seq"].median()
    # 700 epochs x 10 sequences + 14 diagnostic evaluations x 100 sequences, on 16 workers
    n_seq, workers = 700 * 10 + (700 // 50) * 100, 16
    print(f"evaluator share: {n_seq} seq x {s:.3g} s/seq / {workers} workers = "
          f"{n_seq * s / workers:.1f} s (paper: under 9 s)")


def refine_backend() -> None:
    header("refinement-backend check")
    d = json.loads((TABLES / "refine-backend-comparison.json").read_text())
    cells = sorted({(r["n_qubits"], r["lam"]) for r in d["results"]})
    devs = sorted({k for r in d["results"] for k in r["per_device"]})
    print(f"worst relative energy spread across devices: {d['worst_relative_spread']:.1e}")
    print(f"covered (N, lambda): {cells}; devices: {devs}")


def main() -> None:
    onset_table()
    onset_scan_cost()
    floor_length()
    n28_arms()
    pairwise_grid()
    pairwise_vs_reference()
    two_by_two()
    sampler_control()
    showcase()
    refine_backend()


if __name__ == "__main__":
    main()
