from codeevaluation.typing.BagOfProperties import BoP
from codeevaluation.typing.column_types import ID, Success, Error
from codeevaluation.metrics.executionResultMetrics import computeSuccessRate


def test_computeSuccessRate():
    results = BoP[ID, Success, Error].from_json("tests/resources/results.json")

    assert computeSuccessRate(results) == 0.6
