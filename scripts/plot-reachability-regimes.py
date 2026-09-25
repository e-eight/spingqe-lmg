"""Three-regime figure: best error and restart success against sequence length.

The counting bound L >= 2d-2 (L >= N for the collective pool, whose reachable
sector has dimension d = N/2+1) separates three regimes.  Far below the bound a
pool returns its reference-state energy regardless of depth.  Just below it,
accuracy is partial and depends strongly on which tokens were drawn.  Above it,
every restart reaches the numerical floor.  The upper panel shows the best
relative error over 12 random restarts; the lower panel shows the fraction of
those restarts reaching the 1e-10 floor, which is the quantity the onset
definition thresholds.  Single column at lambda = 2.0 (the paper's anchor
coupling); the lambda = 1.5 behavior is qualitatively identical and the text
says so.  The two panels share an x-axis so the counting-bound line can be
traced from the error collapse down to the success-fraction rise.

Data: data/tables/lreach-2026-07-26.csv (356 cells).
CPU-only; runs on a login node.
Output: figures/reachability-regimes.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import spingqe_lmg.analysis.plot_style  # noqa: F401 — applies uniform rcParams
from spingqe_lmg.analysis.plots import POOL_LABEL, VARIANT_STYLE, save_figure

ROOT = Path(__file__).parent.parent
CSV = ROOT / "data" / "tables" / "lreach-2026-07-26.csv"
OUT = ROOT / "figures" / "reachability-regimes.pdf"

N_FIXED = 16
LAMBDAS = (2.0,)
POOLS = ("collective", "pairwise-ext-mf", "pairwise-all")
FLOOR = 1e-15

# The production-sufficient window of Table V, floor length <= L < onset, drawn
# as a band. Its two edges come from DIFFERENT campaigns, which is why the
# caption says so:
#   FLOOR_LENGTH -- smallest L at which all production seeds (train-and-refine,
#     6 seeds) reach the convergence floor; L = 26 for N <= 18. Not a sampled
#     point of this sweep, whose collective rows step 24 -> 28.
#   ONSET -- smallest L at which >= 1/2 of the 12 random restarts plotted here
#     reach the floor; 0.33 at L = 24 and 0.75 at L = 28 put it at 28.
FLOOR_LENGTH = 26
ONSET = 28

# The manuscript's own short forms for these two pools (Tables I and III). Set
# here rather than in POOL_LABEL, which three other plot scripts share. The
# narrower legend is what leaves room for the band label in the upper panel.
SHORT_LABEL = {
    "pairwise-ext-mf": "ext. pair.",
    "pairwise-all": "std. pair.",
}

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--csv", type=Path, default=CSV, help="Canonical reachability CSV")
ap.add_argument("--out", type=Path, default=OUT, help="Output PDF path")
args = ap.parse_args()

df = pd.read_csv(args.csv, comment="#")
df = df[df["n_qubits"] == N_FIXED]

fig, axes = plt.subplots(
    2, len(LAMBDAS), figsize=(3.5, 3.9), sharex=True, sharey="row", squeeze=False
)

for col, lam in enumerate(LAMBDAS):
    ax_err, ax_succ = axes[0][col], axes[1][col]
    sub = df[df["lam"] == lam]
    for pool in POOLS:
        grp = sub[sub["pool"] == pool].sort_values("seq_len")
        if grp.empty:
            continue
        color, marker, ls = VARIANT_STYLE[pool]
        ax_err.semilogy(
            grp["seq_len"],
            grp["best_rel_error"].clip(lower=FLOOR),
            marker=marker,
            color=color,
            linestyle=ls,
            label=SHORT_LABEL.get(pool, POOL_LABEL.get(pool, pool)),
        )
        ax_succ.plot(
            grp["seq_len"],
            grp["success_fraction"],
            marker=marker,
            color=color,
            linestyle=ls,
        )
    # The counting bound for the collective pool at this N, then the
    # production-sufficient window between the floor length and the onset. The
    # window is only two tokens wide against an axis spanning 8 to 64, so it
    # gets a leader line to a label rather than relying on the fill alone.
    for ax in (ax_err, ax_succ):
        ax.axvline(N_FIXED, color="k", linestyle="--", alpha=0.45)
        ax.axvspan(FLOOR_LENGTH, ONSET, color="0.45", alpha=0.20, lw=0, zorder=0)
    # Label the bound line where the panel is empty, left of the line at mid
    # height.  Kept short: the full phrase set rotated is taller than the panel,
    # and the caption already names this as the counting bound.
    ax_err.annotate(
        r"$L = N$",
        xy=(N_FIXED, 0.5),
        xycoords=("data", "axes fraction"),
        xytext=(-4, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        rotation=90,
        fontsize=9,
        color="0.35",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
    )
    # ax_err.annotate(
    #     rf"$\lambda = {lam}$",
    #     xy=(0.97, 0.05),
    #     xycoords="axes fraction",
    #     ha="right",
    #     va="bottom",
    # )
    # NO 0.5 threshold rule here, deliberately. The onset is the smallest
    # SAMPLED L reaching half, which at N = 16 is 28: the collective rows step
    # 24 (4/12) -> 28 (9/12). The segment joining those two samples crosses 0.5
    # at L = 25.6, four tenths of a token from the band's LEFT edge at 26. A
    # rule at 0.5 therefore draws the eye to the floor length, a quantity from
    # the production campaign, and reads as though it located the onset. The
    # band already marks both edges; the rule only mislabels them.

    # Labelled in the upper panel, below the legend, where the shortened pool
    # labels leave room for the phrase on one line. The leader points at the
    # band, which is too narrow at two tokens to carry a label of its own.
    ax_err.annotate(
        "production-sufficient",
        xy=((FLOOR_LENGTH + ONSET) / 2, 3e-10),
        xytext=(31, 3e-10),
        ha="left",
        va="center",
        fontsize=7,
        color="0.30",
        arrowprops={
            "arrowstyle": "->",
            "color": "0.30",
            "lw": 0.8,
            "shrinkA": 1,
            "shrinkB": 1,
        },
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
    )
    ax_succ.set_xlabel(r"sequence length $L$ (tokens)")
    ax_succ.set_ylim(-0.05, 1.05)

axes[0][0].set_ylabel("best relative error")
axes[1][0].set_ylabel("fraction of restarts\nreaching $10^{-10}$")
# The upper panel's mid-band, between the flat pairwise series and the collective
# floor, is the only region either panel leaves empty at every L. Narrowed from
# the default so the box clears the production-sufficient band at L = 26-28;
# at the default width it reached back past L = 18 and covered it.
axes[0][0].legend(loc="center right", fontsize=7, handlelength=1.6, borderpad=0.4)

save_figure(fig, args.out)
