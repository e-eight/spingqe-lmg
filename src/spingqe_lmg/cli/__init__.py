"""CLI entry points — thin argparse wrappers around core library."""

from spingqe_lmg.cli.analysis import (
    aggregate_main,
    exact_main,
    state_quality_main,
    sweep_plan_main,
)
from spingqe_lmg.cli.train import _generate_candidates, refine_main, train_main
from spingqe_lmg.config import MANIFEST_FIXED_COLUMNS, manifest_row_to_overrides, toml_literal

_toml_literal = toml_literal  # backward-compat alias

__all__ = [
    "aggregate_main",
    "exact_main",
    "state_quality_main",
    "sweep_plan_main",
    "refine_main",
    "train_main",
    "_generate_candidates",
    "MANIFEST_FIXED_COLUMNS",
    "manifest_row_to_overrides",
    "_toml_literal",
]
