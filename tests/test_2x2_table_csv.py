"""Tests for scripts/make-2x2-table-csv.py (Task 8, fixes spec F2)."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.script


def _load_module():
    path = Path(__file__).parent.parent / "scripts" / "make-2x2-table-csv.py"
    spec = importlib.util.spec_from_file_location("make_2x2_table_csv", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_2x2_table_csv"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod():
    return _load_module()


def test_median_not_minimum(mod):
    """The F2 regression: an even count must interpolate, never take the min."""
    # min 1.0e-3, median 2.5e-3
    assert mod.median([1.0e-3, 2.0e-3, 3.0e-3, 4.0e-3]) == pytest.approx(2.5e-3, abs=1e-15)


def test_median_of_the_legacy_ext_mf_triple_is_not_the_printed_value(mod):
    """Guards the exact defect found: 2.339e-3 was the min, 2.386e-3 the median."""
    errs = [2.4030e-3, 2.3395e-3, 2.3859e-3]
    assert mod.median(errs) == pytest.approx(2.3859e-3, abs=1e-9)
    assert mod.median(errs) != min(errs)


def test_median_of_odd_count_is_the_middle_value(mod):
    assert mod.median([3.0, 1.0, 2.0]) == pytest.approx(2.0, abs=1e-12)


def test_gain_is_reference_over_refined(mod):
    assert mod.gain(reference=1.0e-1, refined=1.0e-2) == pytest.approx(10.0, abs=1e-9)


def test_gain_of_an_inert_configuration_is_one(mod):
    """A configuration that does nothing must report 1.00x, not 0 or inf."""
    assert mod.gain(reference=2.4493e-3, refined=2.4493e-3) == pytest.approx(1.0, abs=1e-12)


def test_gain_raises_on_zero_refined(mod):
    """A refined error of exactly zero would make the gain meaningless."""
    with pytest.raises(ValueError):
        mod.gain(reference=1.0e-1, refined=0.0)


def test_requires_ten_seeds(mod):
    """A cell with fewer than ten seeds must fail loudly, not silently average."""
    with pytest.raises(ValueError, match="10 seeds"):
        mod.check_seed_count("pairwise-all-mf_lam2.0", n_seeds=3)
    mod.check_seed_count("pairwise-all-mf_lam2.0", n_seeds=10)  # no raise
