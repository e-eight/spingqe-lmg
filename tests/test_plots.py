import pandas as pd
import pytest

from spingqe_lmg.analysis.plots import variant_column
from spingqe_lmg.enums import Variant

pytestmark = pytest.mark.script


def _df(pool_kind, connectivity="all", init_state="zero", paulis=None):
    data = {
        "pool_kind": [pool_kind],
        "connectivity": [connectivity],
        "init_state": [init_state],
    }
    if paulis is not None:
        data["paulis"] = [paulis]
    return pd.DataFrame(data)


def test_collective():
    df = _df("collective")
    assert variant_column(df).iloc[0] == Variant.COLLECTIVE.value


def test_pairwise_all_zero():
    df = _df("pauli_pair", "all", "zero", "XX,YY,ZZ")
    assert variant_column(df).iloc[0] == Variant.PAIRWISE_ALL.value


def test_pairwise_all_ghz_x():
    df = _df("pauli_pair", "all", "ghz_x", "XX,YY,ZZ")
    assert variant_column(df).iloc[0] == Variant.PAIRWISE_ALL.value


def test_pairwise_all_mf_without_ext_words():
    df = _df("pauli_pair", "all", "mean_field", "XX,YY,ZZ")
    assert variant_column(df).iloc[0] == Variant.PAIRWISE_ALL.value


def test_pairwise_ext_mf():
    df = _df("pauli_pair", "all", "mean_field", "XX,YY,ZZ,YZ,ZY")
    assert variant_column(df).iloc[0] == Variant.PAIRWISE_EXT_MF.value


def test_pairwise_ext_mf_all_hp_words():
    df = _df("pauli_pair", "all", "mean_field", "YZ,ZY,XY,YX")
    assert variant_column(df).iloc[0] == Variant.PAIRWISE_EXT_MF.value


def test_pairwise_ext_mf_single_hp_word():
    df = _df("pauli_pair", "all", "mean_field", "XX,YY,ZZ,YZ")
    assert variant_column(df).iloc[0] == Variant.PAIRWISE_EXT_MF.value


def test_pairwise_nn_zero():
    df = _df("pauli_pair", "nn", "zero", "XX,YY,ZZ")
    assert variant_column(df).iloc[0] == Variant.PAIRWISE_NN.value


def test_pairwise_nn_mean_field_no_ext_words():
    df = _df("pauli_pair", "nn", "mean_field", "XX,YY,ZZ")
    assert variant_column(df).iloc[0] == Variant.PAIRWISE_NN.value


def test_pairwise_nn_mean_field_with_ext_words():
    df = _df("pauli_pair", "nn", "mean_field", "XX,YY,ZZ,XY")
    assert variant_column(df).iloc[0] == Variant.PAIRWISE_EXT_MF.value


def test_column_name_preserved():
    df = _df("pauli_pair")
    result = variant_column(df)
    assert result.name == "pool_kind"


@pytest.mark.parametrize(
    "pool_kind,connectivity,init_state,paulis,expected",
    [
        ("collective", "all", "zero", "", Variant.COLLECTIVE.value),
        ("collective", "all", "mean_field", "", Variant.COLLECTIVE.value),
        ("pauli_pair", "all", "zero", "XX,YY,ZZ", Variant.PAIRWISE_ALL.value),
        ("pauli_pair", "all", "ghz_x", "XX,YY,ZZ", Variant.PAIRWISE_ALL.value),
        ("pauli_pair", "all", "mean_field", "XX,YY,ZZ", Variant.PAIRWISE_ALL.value),
        ("pauli_pair", "all", "mean_field", "XX,YY,ZZ,YZ", Variant.PAIRWISE_EXT_MF.value),
        ("pauli_pair", "nn", "zero", "XX,YY,ZZ", Variant.PAIRWISE_NN.value),
        ("pauli_pair", "nn", "mean_field", "XX,YY,ZZ,YZ", Variant.PAIRWISE_EXT_MF.value),
    ],
)
def test_parametrized(pool_kind, connectivity, init_state, paulis, expected):
    df = _df(pool_kind, connectivity, init_state, paulis)
    assert variant_column(df).iloc[0] == expected


def test_legacy_no_paulis_column():
    df = pd.DataFrame(
        {
            "pool_kind": ["pauli_pair"],
            "connectivity": ["all"],
            "init_state": ["mean_field"],
        }
    )
    assert variant_column(df).iloc[0] == Variant.PAIRWISE_EXT_MF.value


def test_no_init_state_column():
    df = pd.DataFrame(
        {
            "pool_kind": ["pauli_pair"],
            "connectivity": ["all"],
        }
    )
    assert variant_column(df).iloc[0] == Variant.PAIRWISE_ALL.value


def test_empty_paulis_string():
    df = _df("pauli_pair", "all", "mean_field", "")
    assert variant_column(df).iloc[0] == Variant.PAIRWISE_ALL.value


def test_multiple_rows_mixed():
    df = pd.DataFrame(
        {
            "pool_kind": ["collective", "pauli_pair", "pauli_pair", "pauli_pair"],
            "connectivity": ["all", "all", "nn", "all"],
            "init_state": ["zero", "zero", "zero", "mean_field"],
            "paulis": ["", "XX,YY,ZZ", "XX,YY,ZZ", "XX,YY,ZZ,YZ"],
        }
    )
    result = variant_column(df)
    assert list(result) == [
        Variant.COLLECTIVE.value,
        Variant.PAIRWISE_ALL.value,
        Variant.PAIRWISE_NN.value,
        Variant.PAIRWISE_EXT_MF.value,
    ]
