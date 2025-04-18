from codeevaluation.typing.BagOfProperties import BagOfPropertiesFactory
from codeevaluation.typing.types import id, success, error
from codeevaluation.metrics.executionResultMetrics import computeSuccessRate


def test_computeSuccessRate():
    results = BagOfPropertiesFactory[id, success, error].from_json(
        "tests/resources/results.json"
    )

    assert computeSuccessRate(results) == 0.6
