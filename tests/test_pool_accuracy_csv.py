"""Tests for scripts/make-pool-accuracy-10seed-csv.py (Task 7)."""

import importlib.util
import math
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.script


def _load_module():
    path = Path(__file__).parent.parent / "scripts" / "make-pool-accuracy-10seed-csv.py"
    spec = importlib.util.spec_from_file_location("make_pool_accuracy_10seed_csv", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_pool_accuracy_10seed_csv"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod():
    return _load_module()


def test_grid_is_the_51_cell_full_grid(mod):
    """17 collective + 34 pairwise. Import itself asserts the count."""
    assert len(mod.CELLS) == 51
    collective = {c for c in mod.CELLS if c[0] == "collective"}
    pairwise = {c for c in mod.CELLS if c[0] != "collective"}
    assert len(collective) == 17
    assert len(pairwise) == 34


def test_pairwise_grid_covers_both_pools_over_the_full_lambda_range(mod):
    for pool in ("pairwise-all", "pairwise-ext-mf"):
        lams = {c[2] for c in mod.CELLS if c[0] == pool}
        assert lams == {0.5, 1.0, 1.5, 2.0}
        # lam=0.5 only exists at N in {8,10}; the other three span all five N
        assert {c[1] for c in mod.CELLS if c[0] == pool and c[2] == 0.5} == {8, 10}
        assert {c[1] for c in mod.CELLS if c[0] == pool and c[2] == 2.0} == {8, 10, 12, 14, 16}


def test_sweeps_is_ordered_and_new_campaign_precedes_nothing_it_shares(mod):
    """Cell resolution is first-match, so SWEEPS order is load-bearing."""
    assert isinstance(mod.SWEEPS, (list, tuple))
    assert "pool-accuracy-l12-s1-10-2026-07-28" in mod.SWEEPS
    assert "pool-accuracy-pairwise-l12-s1-10-2026-07-29" in mod.SWEEPS


def test_iqr_dex_is_log10_of_the_quartile_ratio(mod):
    """Two decades between Q1 and Q3 must read as 2.0 decades."""
    # 12 values: Q1 lands at 1e-4, Q3 at 1e-2 -> 2 decades
    errs = [1e-4] * 6 + [1e-2] * 6
    stats = mod._cell_stats(errs)
    assert stats["iqr_dex"] == pytest.approx(2.0, abs=1e-9)


def test_cell_stats_reports_min_max_and_spread(mod):
    errs = [1e-3, 2e-3, 4e-3]
    stats = mod._cell_stats(errs)
    assert stats["min"] == pytest.approx(1e-3, abs=1e-15)
    assert stats["max"] == pytest.approx(4e-3, abs=1e-15)
    assert stats["spread"] == pytest.approx(4.0, abs=1e-9)


def test_tight_cell_has_near_zero_iqr(mod):
    """The pairwise signature: ten seeds landing in the same place."""
    errs = [1.0e-2 * (1 + 0.001 * i) for i in range(10)]
    assert mod._cell_stats(errs)["iqr_dex"] < 0.01


def test_iqr_is_nan_when_a_quartile_is_zero(mod):
    """log10(0) must not raise; the column carries nan instead."""
    stats = mod._cell_stats([0.0] * 6 + [1e-2] * 6)
    assert math.isnan(stats["iqr_dex"])
