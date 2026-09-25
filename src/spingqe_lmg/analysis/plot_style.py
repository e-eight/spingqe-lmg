"""Centralised matplotlib style for all SpinGQE-LMG publication figures.

Import this module before ``import matplotlib.pyplot as plt`` to apply
uniform font sizes, line widths, and marker sizes across all figure scripts.
"""

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    }
)
