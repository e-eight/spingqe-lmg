"""Plotting helpers (matplotlib, headless-safe)."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from spingqe_lmg.analysis import plot_style  # noqa: F401 — applies uniform rcParams
from spingqe_lmg.enums import InitState, PoolKind, Variant
from spingqe_lmg.exact import exact_qpt_curve

# Words that distinguish the extended pairwise pool from standard pairwise.
_EXTENDED_PAULI_WORDS = frozenset({"YZ", "ZY", "XY", "YX"})

# CVD-safe color palette with redundant shape/line-style encoding so that
# figures remain readable in greyscale and for readers with colour-vision
# deficiency. Headline pools use high-contrast blue/orange; control pools
# are de-emphasized in greys.
VARIANT_STYLE = {
    Variant.COLLECTIVE.value: ("#006BA4", "D", "-"),
    Variant.PAIRWISE_EXT_MF.value: ("#FF800E", "^", "-."),
    Variant.PAIRWISE_ALL.value: ("#595959", "o", "--"),
    Variant.PAIRWISE_NN.value: ("#ABABAB", "s", ":"),
}
DEFAULT_STYLE = ("#898989", "x", "-")

# Floor for log-scale error plots — clip zero/near-zero values so semilogy
# doesn't choke, without visually distorting genuinely tiny errors.
CLIP_LOW = 1e-16

# Evaluator-specific styling for benchmark/crossover figures.
# Keys match the backend identifiers used in benchmark CSV output.
# All colors from tableau-colorblind10 (CVD policy, style-guide §6); CPU/GPU
# tiers of the same custom evaluator share color (same family) but differ in
# both marker and linestyle, so color loss (greyscale, CVD) never leaves two
# series distinguishable by only one channel. No two entries below share the
# same (marker, linestyle) pair, even across families/colors -- that pairing,
# not the marker or linestyle alone, is what must stay unique.
EVALUATOR_STYLE = {
    "lightning.qubit": ("#595959", "o", "--"),
    "lightning.gpu": ("#C85200", "s", "-"),
    "lightning.qubit (grouped)": ("#595959", "v", ":"),
    "lightning.gpu (grouped)": ("#C85200", "^", "-."),
    "cudaq_v1": ("#006BA4", "D", "-."),
    "incremental": ("#5F9ED1", "P", "-"),  # medium-blue pentagon (CPU tier)
    "incremental_gpu": ("#5F9ED1", "*", "-."),  # medium-blue star, dash-dot (GPU tier)
    "parity_even": ("#ABABAB", "h", "-"),  # light-gray hexagon (CPU tier)
    "parity_even_gpu": ("#ABABAB", "X", "-."),  # light-gray X, dash-dot (GPU tier)
    "dicke": ("#FF800E", "^", "--"),
}

SCALING_STYLE = {
    "lightning": ("#595959", "o", "--"),
    "incremental": ("#006BA4", "D", "-"),
}

# Prose names for pool variants used in figure labels and manuscript text.
# Code identifiers (pairwise-all, pairwise-ext-mf, ...) must not appear in figures.
POOL_LABEL = {
    Variant.COLLECTIVE.value: "collective",
    Variant.PAIRWISE_EXT_MF.value: "extended pairwise",
    Variant.PAIRWISE_ALL.value: "standard pairwise",
    Variant.PAIRWISE_NN.value: "nearest-neighbor pairwise",
}

# ---------------------------------------------------------------------------
# Shared utilities for plot scripts
# ---------------------------------------------------------------------------


def save_figure(fig: plt.Figure, path: str | Path) -> None:
    """Standard figure output: tight-layout crop, save, close, and log.

    Consolidates the tight_layout/savefig/close/print sequence every plot
    script repeats.
    """
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def compute_shared_ylim(
    values: Sequence[float],
    clip_low: float = CLIP_LOW,
    margin: float = 2.0,
) -> tuple[float, float]:
    """Return (y_min, y_max) that spans all *values* with symmetric log-scale padding.

    Used by multiple plot scripts to give subfigures a uniform y-axis range
    so visual comparisons across panels are fair.
    """
    if not values:
        return (1e-5, 1.0)
    vmin = max(min(values) * 0.5, clip_low)
    vmax = max(values) * margin
    return (vmin, vmax)


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------


def variant_column(df: pd.DataFrame) -> pd.Series:
    name = (
        df["pool_kind"]
        .str.replace(PoolKind.PAULI_PAIR.value, "pairwise")
        .str.cat(df["connectivity"], sep="-")
    )
    name = name.where(df["pool_kind"] != PoolKind.COLLECTIVE.value, Variant.COLLECTIVE.value)
    # Extended pairwise uses mean-field reference + expanded Pauli word set
    # (at least one HP word: YZ, ZY, XY, YX). Fall back to the old
    # init_state-only heuristic when the paulis column is absent (legacy
    # CSVs generated before the column was added).
    if "init_state" in df.columns:
        is_mean_field = (df["pool_kind"] == PoolKind.PAULI_PAIR.value) & (
            df["init_state"] == InitState.MEAN_FIELD.value
        )
        if "paulis" in df.columns and df["paulis"].notna().any():
            pauli_sets = df["paulis"].fillna("").str.split(",")
            has_ext = pauli_sets.apply(lambda words: bool(set(words) & _EXTENDED_PAULI_WORDS))
            is_ext = is_mean_field & has_ext
        else:
            is_ext = is_mean_field
        name = name.where(~is_ext, Variant.PAIRWISE_EXT_MF.value)
    return name


def plot_exact_curves(
    n_qubits: Sequence[int],
    h: float,
    lams: np.ndarray,
    gamma: float,
    out_path: str | Path,
) -> None:
    """Exact LMG QPT curves: ground-state energy per spin vs lambda/h.

    Uses the (N+1)-dimensional permutation-symmetric (J=N/2) Dicke sector,
    which gives the exact ground-state energy at polynomial cost in N.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(n_qubits)))
    linestyles = ["-", "--", "-.", ":"]
    for k, n in enumerate(n_qubits):
        curve = exact_qpt_curve(n, h, lams, gamma)
        ax.plot(
            lams / h,
            curve / n,
            color=colors[k],
            linestyle=linestyles[k % len(linestyles)],
            label=rf"$N = {n}$",
        )
    ax.axvline(1.0, color="k", linestyle="--", alpha=0.4, label=r"$\lambda_c = h$")
    ax.set_xlabel(r"$\lambda / h$")
    ax.set_ylabel(r"Ground-state energy per spin, $E_0/N$")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_qpt_errors(df: pd.DataFrame, out_path: str | Path) -> None:
    """Best-over-seeds energy error vs lam/h: rows = (raw, refined), columns = N."""
    df = df.copy()
    df["lam_over_h"] = df["lam"] / df["h"]
    df["variant"] = variant_column(df)
    ns = sorted(df["n_qubits"].unique())
    has_refined = "refined_abs_error" in df and df["refined_abs_error"].notna().any()
    metrics = [("abs_error", "raw GQE")] + (
        [("refined_abs_error", "refined")] if has_refined else []
    )

    fig, axes = plt.subplots(len(metrics), len(ns), figsize=(18, 4 * len(metrics)), squeeze=False)
    for row, (metric, metric_label) in enumerate(metrics):
        for col, n in enumerate(ns):
            ax = axes[row][col]
            sub = df[df["n_qubits"] == n]
            for variant, grp in sub.groupby("variant"):
                color, marker, ls = VARIANT_STYLE.get(variant, DEFAULT_STYLE)
                best = grp.groupby("lam_over_h")[metric].min()
                ax.semilogy(
                    best.index,
                    np.maximum(best.values, 1e-12),
                    marker=marker,
                    color=color,
                    linestyle=ls,
                    label=POOL_LABEL.get(variant, variant),
                )
            ax.axvline(1.0, color="k", linestyle="--", alpha=0.35)
            if row == 0:
                ax.set_title(rf"$N = {n}$")
            if row == len(metrics) - 1:
                ax.set_xlabel(r"$\lambda / h$")
            if col == 0:
                ax.set_ylabel(rf"{metric_label}: $(E - E_0)/h$")
            if row == 0 and col == 0:
                ax.legend()
    fig.suptitle("GQE energy error across the LMG QPT (best over seeds)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
