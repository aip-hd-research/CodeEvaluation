import pytest
import polars as pl
from typing import Type

from codeevaluation.typing.BagOfProperties import (
    BagOfProperties,
    castBoPtype,
    BagOfPropertiesFactory,
)


# === Test Types === #
class id:
    datatype: Type = int


class codeJava:
    datatype: Type = str


class status:
    datatype: Type = str


class anotherType:
    datatype: Type = int


# Sample data
testdata = [
    {"id": 1, "codeJava": "A"},
    {"id": 2, "codeJava": "B"},
]

extradata = [
    {"id": 1, "codeJava": "A", "status": "OK"},
    {"id": 2, "codeJava": "B", "status": "FAIL"},
]

incomplete_data = [
    {"id": 1},  # missing codeJava
    {"codeJava": "A"},  # missing id
]


# === Basic: Type matching === #
def test_type_args_preserved():
    boP = BagOfPropertiesFactory[id, codeJava].from_dicts(testdata)
    expected = (id, codeJava)
    actual = getattr(boP.__class__, "__type_args__", None)
    assert actual == expected


# === Schema generation === #
def test_schema_matches_types():
    boP = BagOfPropertiesFactory[id, codeJava].from_dicts(testdata)
    expected_schema = {"id": int, "codeJava": str}
    assert boP.schema == expected_schema


# === Dataframe contents === #
def test_dataframe_content():
    boP = BagOfPropertiesFactory[id, codeJava].from_dicts(testdata)
    df = boP.df
    assert isinstance(df, pl.DataFrame)
    assert df.shape == (2, 2)
    assert df.columns == ["id", "codeJava"]
    assert df[0, "id"] == 1
    assert df[1, "codeJava"] == "B"


def test_factory_new_creates_empty_bop():
    bop = BagOfPropertiesFactory[id, codeJava].new()

    # Check the type args are correctly stored
    expected_types = (id, codeJava)
    actual_types = getattr(bop.__class__, "__type_args__", None)
    assert actual_types == expected_types

    # Check that the dataframe is empty but has the right schema
    assert bop.df.shape == (0, 2)
    assert bop.df.columns == ["id", "codeJava"]

    # Check that schema matches expected Python types
    assert bop.schema == {"id": int, "codeJava": str}


# === Incomplete rows handled === #
def test_missing_keys_handled():
    boP = BagOfPropertiesFactory[id, codeJava].from_dicts(incomplete_data)
    assert boP.df.shape == (2, 2)
    assert boP.df.null_count().sum_horizontal().sum() > 0  # some nulls expected


# === Decorator casting === #
@castBoPtype
def func_expects_codeJava_only(bop: BagOfProperties[id, codeJava]):
    return bop.df.columns


def test_castBoPtype_casts_correctly():
    bop_full = BagOfPropertiesFactory[id, codeJava, status].from_dicts(extradata)
    result = func_expects_codeJava_only(bop_full)
    assert result == ["id", "codeJava"]


# === Decorator skips if exact match === #
@castBoPtype
def func_expects_full(bop: BagOfProperties[id, codeJava, status]):
    return bop.df.columns


def test_castBoPtype_noop_on_exact_match():
    bop = BagOfPropertiesFactory[id, codeJava, status].from_dicts(extradata)
    assert func_expects_full(bop) == ["id", "codeJava", "status"]


# === Edge case: missing datatype === #
def test_missing_datatype_raises():
    class badType:
        pass  # missing 'datatype'

    with pytest.raises(TypeError):
        BagOfPropertiesFactory[id, badType].from_dicts(testdata)


# === File load errors === #
def test_bad_json_raises(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not a json")
    with pytest.raises(ValueError, match="Error loading JSON file"):
        BagOfPropertiesFactory[id, codeJava].from_json(str(bad_file))


def test_non_list_json_raises(tmp_path):
    bad_file = tmp_path / "notalist.json"
    bad_file.write_text('{"id": 1}')
    with pytest.raises(
        ValueError, match="JSON file must contain a list of dictionaries"
    ):
        BagOfPropertiesFactory[id, codeJava].from_json(str(bad_file))


# === CSV loading === #
def test_csv_loading(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("id,codeJava\n1,A\n2,B\n")
    bop = BagOfPropertiesFactory[id, codeJava].from_csv(str(csv_file))
    assert bop.df.shape == (2, 2)
    assert bop.df.columns == ["id", "codeJava"]
