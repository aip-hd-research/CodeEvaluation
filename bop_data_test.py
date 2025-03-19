import pytest

from bop import BoP, castBoPtype, StringColumn, IntegerColumn, BoolColumn

# === Result Types === #
class Correct(IntegerColumn):
    """For correct column of the results.json"""
    name = "correct"

class Total(IntegerColumn):
    """For total column of the results.json"""
    name = "total"

class Success(BoolColumn):
    """For success column of the results.json"""
    name = "success"

class Error(StringColumn):
    """For error column of the results.json"""
    name = "error"

class Path(StringColumn):
    """For path column of the results.json"""
    name = "path"

class Description(StringColumn):
    """For description column of the results.json"""
    name = "description"


results = BoP[Path, Success, Error,Correct, Total, Description].from_json("results.json")
results.show()

results_cut = BoP[Path, Success, Error].from_json("results.json")
results_cut.show()
