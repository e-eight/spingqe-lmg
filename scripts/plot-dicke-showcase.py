"""Dicke-sector collective-GQE showcase: LMG ground state up to N=1000.

Permutation-invariant (collective) circuits stay in the (N+1)-dimensional
symmetric subspace, so the GQE search and its energy evaluation are classically
simulable at poly cost in N (Anschuetz et al., Quantum 7, 1189 (2023); Somma
et al., PRL 97, 190501 (2006)). This figure is the large-N demonstration in the
broken phase (lam/h = 1.5), N = 50 -> 1000.

Two arms are plotted, both at the fine (8/N-rescaled) vocabulary, 10 seeds per
N, each as best + median with a light band between them:
- mean-field reference: 1.7e-4 -> 2.1e-6 median, best to 1.1e-7 at N=1000,
- symmetric |0...0> reference: 2.7e-3 -> 2.2e-2 median.
Holding the vocabulary fixed is what makes this a clean reference-state
comparison. Dashed lines are the reference-state energies with no search at
all: mean-field (descending ~1e-3 -> 5e-5) and zero (~7.5e-2, flat). The worst
seeds of the mean-field arm collapse exactly onto its reference line, so those
dashes are also the failure floor.

The zero-reference campaign also holds unscaled-vocabulary runs (``N*-c-s*``);
the angle vocabulary is not an axis the paper analyses, so they get no series.

Data: data/runs/dicke-showcase-{mf,zero}-seeds-2026-08-06
(lam=1.5, gamma=0; 10 seeds per (N, arm)). Cells with fewer than MIN_SEEDS
finished runs are dropped and reported, so a campaign still in flight cannot
put a two-seed "median" on the page.
CPU-only aggregation + plot; runs on a login node.
Output: figures/dicke-showcase.pdf
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import spingqe_lmg.analysis.plot_style  # noqa: F401 — applies uniform rcParams
from spingqe_lmg.analysis.aggregate import aggregate_sweep
from spingqe_lmg.analysis.plots import save_figure

ROOT = Path(__file__).parent.parent

CAMPAIGNS = [
    ROOT / "data" / "runs" / "dicke-showcase-mf-seeds-2026-08-06",
    ROOT / "data" / "runs" / "dicke-showcase-zero-seeds-2026-08-06",
]

FINE = 8.0
COARSE = 0.0

# U+2009 THIN SPACE, so the figure renders "m. f." the same way the manuscript's
# m.\,f. does. Verified present in DejaVu Sans, the font these figures actually
# resolve to. Mathtext ($\mathrm{m.\,f.}$) would give a glyph-free kern instead,
# but it renders 13% wider and leaves a visible gap at the math/text boundary.
MF = "m.\u2009f."

# Only the fine-token arms are plotted. Holding the vocabulary fixed is what
# makes this a clean reference-state comparison; the coarse arms were run to
# establish that (they give the same ordering with far larger scatter) and are
# reported qualitatively in the text rather than drawn here, since the angle
# vocabulary is not an axis this paper analyses.
# (label, colour, marker, linestyle) -- colour, marker and linestyle all carry
# the reference state, so no single channel is load-bearing.
# Labels omit "ref." to keep the legend box narrow, since it sits over the
# mean-field reference-energy line; the caption says the labels are reference
# states and names which baseline is which.
SERIES = {
    ("mean_field", FINE): ("mean-field", "#006BA4", "D", "-"),
    ("zero", FINE): ("zero", "#595959", "o", "--"),
}

# The two no-search reference baselines carry no legend entries; the caption
# explains them. Their linestyles must avoid the series linestyles above, so
# neither can be "-" or "--".
BASELINES = {
    "mean_field": ":",
    "zero": "-.",
}

N_SERIES = (50, 100, 200, 500, 1000)
Y_LIMS = (1e-7, 2e-1)
BASELINE_COLOR = "0.35"

# A cell with only a couple of finished seeds is not a median, it is whichever
# seeds happened to land first. Campaigns run 10 seeds; drop anything under
# this and say so, rather than drawing a point the reader cannot discount.
MIN_SEEDS = 8


def _per_n(errs: dict[int, np.ndarray], n: int) -> tuple[float, float]:
    vals = errs[n]
    return float(vals.min()), float(np.median(vals))


def _reference_rel_error(run_dir: Path, n_qubits: int) -> float | None:
    """Reference-only relative error from a campaign run's metadata."""
    meta = json.loads((run_dir / "metadata.json").read_text())
    if "initial_energy" not in meta:
        return None
    return abs(meta["initial_energy"] - meta["ground_energy"]) / abs(meta["ground_energy"])


def _baselines(campaign: Path, n_qubits: int, init_state: str) -> float:
    values = []
    for run_dir in campaign.iterdir():
        if not run_dir.is_dir() or not (run_dir / "config.json").exists():
            continue
        cfg = json.loads((run_dir / "config.json").read_text())
        if cfg["hamiltonian"]["n_qubits"] != n_qubits:
            continue
        if cfg["train"].get("init_state", "zero") != init_state:
            continue
        rel = _reference_rel_error(run_dir, n_qubits)
        if rel is not None:
            values.append(rel)
    return float(np.median(values))


def main() -> None:
    frames = []
    for campaign in CAMPAIGNS:
        df, incomplete = aggregate_sweep(campaign)
        for name in incomplete:
            print(f"skip incomplete: {name}")
        frames.append(df)
    import pandas as pd

    df = pd.concat(frames, ignore_index=True)
    print(f"runs: {len(df)}; N = {sorted(df['n_qubits'].unique())}")

    errs: dict[tuple[str, float], dict[int, np.ndarray]] = {}
    for key in SERIES:
        init_state, scale = key
        sub = df[(df["init_state"] == init_state) & (df["angle_scale"] == scale)]
        errs[key] = {
            n: np.sort(g["refined_rel_error"].to_numpy()) for n, g in sub.groupby("n_qubits")
        }

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    # Captured for the direct labels below: every line on the graphic gets a
    # word next to it, so no encoding is caption-only.
    drawn: dict[str, dict[str, np.ndarray]] = {}
    for key, (label, color, marker, ls) in SERIES.items():
        cell = errs[key]
        # Only cells with enough finished seeds to make a median meaningful.
        thin = {n: len(cell[n]) for n in N_SERIES if n in cell and len(cell[n]) < MIN_SEEDS}
        ns = [n for n in N_SERIES if n in cell and len(cell[n]) >= MIN_SEEDS]
        if thin:
            print(f"under-populated (<{MIN_SEEDS} seeds), dropped: {key}, N={thin}")
        if not ns:
            print(f"no usable data: {key}")
            continue
        if len(ns) < len(N_SERIES):
            print(f"partial: {key}; missing N={[n for n in N_SERIES if n not in ns]}")
        bests, medians = zip(*(_per_n(cell, n) for n in ns), strict=True)
        x = np.asarray(ns)
        bests = np.asarray(bests)
        medians = np.asarray(medians)
        ax.semilogy(x, medians, color=color, marker=marker, linestyle=ls, label=label)
        ax.semilogy(x, bests, color=color, marker=marker, linestyle=ls, alpha=0.45, lw=0.9)
        ax.fill_between(x, bests, medians, color=color, alpha=0.12, interpolate=True)
        drawn[key[0]] = {"x": x, "median": medians, "best": bests, "color": color}

    base_y: dict[str, np.ndarray] = {}
    for init_state, ls in BASELINES.items():
        campaign = CAMPAIGNS[1] if init_state == "zero" else CAMPAIGNS[0]
        y = [_baselines(campaign, n, init_state) for n in N_SERIES]
        base_y[init_state] = np.asarray(y)
        ax.semilogy(
            np.asarray(N_SERIES),
            np.asarray(y),
            color=BASELINE_COLOR,
            linestyle=ls,
            lw=1.0,
        )

    ax.set_xscale("log")
    ax.set_xticks(list(N_SERIES))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(*Y_LIMS)
    ax.set_xlabel(r"$N$ (spins)")
    ax.set_ylabel("post-refinement relative error")
    ax.grid(alpha=0.25, which="both")

    # Direct labels instead of a legend box. The box sat over the mean-field
    # reference line and the zero arm's best-of-ten line, and it named only the
    # two reference states -- leaving line weight, the band, and both baselines
    # to the caption. Every line now carries a word next to it, which is what
    # the design asks for ("words directly on the graphic beat caption
    # archaeology"). Positions are taken from the drawn arrays, so they follow
    # the data if a campaign is re-aggregated.
    # A near-opaque white box behind each label so it stays readable where it
    # crosses a line or a grid rule. MF rather than "mean-field" keeps the
    # two longest labels clear of neighbouring series.
    lab = {
        "fontsize": 7,
        "textcoords": "offset points",
        "bbox": {
            "boxstyle": "round,pad=0.15",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.8,
        },
    }

    def at(x, y):
        return {"xy": (float(x), float(y))}

    # Arm labels sit against their own median line. The zero arm is labelled at
    # the left edge so it cannot be mistaken for the zero baseline, which is
    # labelled at N=200 further up.
    for init_state, name, idx, dy, ha in (
        ("mean_field", MF, 2, 9, "center"),
        ("zero", "zero", 0, 9, "left"),
    ):
        d = drawn.get(init_state)
        if d is None:
            continue
        ax.annotate(
            name,
            **at(d["x"][idx], d["median"][idx]),
            xytext=(4 if ha == "left" else 0, dy),
            ha=ha,
            color=d["color"],
            **lab,
        )

    # Line weight and the band, explained once on the arm where the median and
    # the best seed are furthest apart.
    mf = drawn.get("mean_field")
    if mf is not None:
        ax.annotate(
            "median",
            **at(mf["x"][-1], mf["median"][-1]),
            xytext=(-4, 10),
            ha="right",
            color=mf["color"],
            **lab,
        )
        ax.annotate(
            "best of 10",
            **at(mf["x"][-1], mf["best"][-1]),
            xytext=(-4, 6),
            ha="right",
            color=mf["color"],
            **lab,
        )

    # Baseline labels: below the flat zero line (above it would clip the axes),
    # and above the descending mean-field line at N=500, the one gap where it
    # clears both the zero arm's best-of-ten line and its own markers.
    for init_state, name, idx, dy in (
        ("zero", "zero only", 2, -8),
        ("mean_field", f"{MF} only", 3, 7),
    ):
        if init_state not in base_y:
            continue
        ax.annotate(
            name,
            **at(N_SERIES[idx], base_y[init_state][idx]),
            xytext=(0, dy),
            ha="center",
            color=BASELINE_COLOR,
            **lab,
        )

    print()
    save_figure(fig, ROOT / "figures" / "dicke-showcase.pdf")


if __name__ == "__main__":
    main()
