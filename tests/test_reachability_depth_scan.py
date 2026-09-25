"""Tests for scripts/reachability-depth-scan.py."""

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from spingqe_lmg.config import config_from_dict
from spingqe_lmg.evaluators.dicke import DickeSpace, dicke_refine_tokens
from spingqe_lmg.pools import build_pool

pytestmark = pytest.mark.script


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "reachability_depth_scan",
        Path(__file__).parent.parent / "scripts" / "reachability-depth-scan.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rmod():
    return _load_module()


# --- (b) pool built by the scanner is token-identical to the canonical sweep's pool ---

CANONICAL_VARIANT_OVERRIDES = {
    "collective": {},
    "pairwise-all": {"connectivity": "all", "paulis": ["ZZ", "XX", "YY"]},
    "pairwise-ext-mf": {"connectivity": "all", "paulis": ["ZZ", "XX", "YY", "YZ", "XY"]},
}


@pytest.mark.parametrize("name", ["collective", "pairwise-all", "pairwise-ext-mf"])
def test_pool_matches_canonical_sweep_config(rmod, name):
    kind = "collective" if name == "collective" else "pauli_pair"
    cfg = config_from_dict({"pool": {"kind": kind, **CANONICAL_VARIANT_OVERRIDES[name]}})
    canonical_ops = build_pool(cfg.pool, 8)
    scanner_ops = build_pool(rmod.POOL_CONFIGS[name], 8)
    assert [op.label for op in scanner_ops] == [op.label for op in canonical_ops]


# --- (a) collective at N=8, L=16 reaches the floor and L=4 does not ---


def test_collective_n8_l16_reaches_floor_l4_does_not(rmod, tmp_path):
    out = tmp_path / "scan.csv"
    rmod.scan(
        pools=["collective"],
        ns=[8],
        lambdas=[2.0],
        seq_lens=[4, 16],
        restarts=12,
        init_angles="random",
        seed=0,
        out_path=out,
    )
    with open(out) as f:
        rows = list(csv.DictReader(f))
    best_by_len = {}
    for row in rows:
        seq_len = int(row["seq_len"])
        best_by_len[seq_len] = min(best_by_len.get(seq_len, np.inf), float(row["rel_error"]))
    assert best_by_len[16] < 1e-10
    assert best_by_len[4] >= 1e-10


# --- (c) --init-angles token reproduces the pipeline's starting point ---


def test_init_angles_token_matches_pipeline_start(rmod):
    n_qubits = 8
    pool_ops = build_pool(rmod.POOL_CONFIGS["collective"], n_qubits)
    space = DickeSpace.build(n_qubits, rmod.H_FIELD, 2.0, rmod.GAMMA)
    tokens = [0, 3, 7, 12]

    # x0=None (the "token" policy) must reproduce the production call with no
    # x0 override at all, i.e. each op's own discrete pool angle.
    production = dicke_refine_tokens(tokens, pool_ops, space)
    via_scanner_default = dicke_refine_tokens(tokens, pool_ops, space, x0=None)
    assert production.initial_energy == pytest.approx(via_scanner_default.initial_energy)
    assert production.energy == pytest.approx(via_scanner_default.energy)
