"""Tests for scripts/table-pool-accuracy.py."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.script


def _load_module():
    path = Path(__file__).parent.parent / "scripts" / "table-pool-accuracy.py"
    spec = importlib.util.spec_from_file_location("table_pool_accuracy", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["table_pool_accuracy"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def tpa():
    return _load_module()


def _seeds_disagree_df():
    # Three seeds at (N=10, lam=1.0, collective): min=1e-6, median=5e-6, max=9e-6.
    rows = [
        {
            "n_qubits": 10, "lam": 1.0, "variant": "collective", "seed": s,
            "refined_rel_error": v, "pool_size": 30,
        }
        for s, v in zip([1, 2, 3], [9e-6, 5e-6, 1e-6], strict=True)
    ]
    return pd.DataFrame(rows)


def test_best_over_seeds_default_is_min(tpa):
    df = _seeds_disagree_df()
    out = tpa.best_over_seeds(df)
    assert out.loc[0, "refined_rel_error"] == pytest.approx(1e-6)


def test_best_over_seeds_median_stat_differs_from_best(tpa):
    df = _seeds_disagree_df()
    best = tpa.best_over_seeds(df, stat="best")
    median = tpa.best_over_seeds(df, stat="median")
    assert best.loc[0, "refined_rel_error"] == pytest.approx(1e-6)
    assert median.loc[0, "refined_rel_error"] == pytest.approx(5e-6)
    assert median.loc[0, "refined_rel_error"] != best.loc[0, "refined_rel_error"]


def _floor_df():
    # Four seeds at (N=8, lam=2.0, collective): two at/below a 1e-10 floor.
    rows = [
        {
            "n_qubits": 8, "lam": 2.0, "variant": "collective", "seed": s,
            "refined_rel_error": v, "rel_error": v, "pool_size": 16,
        }
        for s, v in zip([1, 2, 3, 4], [3e-13, 8e-11, 4e-9, 2e-4], strict=True)
    ]
    return pd.DataFrame(rows)


def test_floor_hits_counts_seeds_at_or_below_floor(tpa):
    out = tpa.best_over_seeds(_floor_df(), floor=1e-10)
    assert out.loc[0, "floor_hits"] == 2


def test_floor_hits_tracks_the_floor_argument(tpa):
    df = _floor_df()
    assert tpa.best_over_seeds(df, floor=1e-8).loc[0, "floor_hits"] == 3
    assert tpa.best_over_seeds(df, floor=1e-14).loc[0, "floor_hits"] == 0


def test_fmt_does_not_clip_below_1e8(tpa):
    # The collective pool's best-over-seeds error reaches 3e-13; a clip at 1e-8
    # would collapse the whole daggered end of the table into one bucket.
    assert "3.2e-13" in tpa._fmt(3.17e-13)
    assert "<" not in tpa._fmt(3.17e-13)


def test_collective_only_layout_daggers_floor_cells(tpa):
    reduced = tpa.best_over_seeds(_floor_df(), stat="median", floor=1e-10)
    out = tpa.to_latex(reduced, targets=[(8, 2.0)], collective_only=True)
    assert out.count("&") == 3  # four columns: N, lambda, central, best
    assert r"\(^\dagger\)" in out


def test_collective_only_omits_dagger_when_no_seed_reaches_floor(tpa):
    reduced = tpa.best_over_seeds(_floor_df(), stat="median", floor=1e-14)
    out = tpa.to_latex(reduced, targets=[(8, 2.0)], collective_only=True)
    assert r"\dagger" not in out
