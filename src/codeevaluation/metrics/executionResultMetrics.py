from codeevaluation.typing.BagOfProperties import BoP
from codeevaluation.typing.column_types import ID, Success


def computeSuccessRate(results: BoP[ID, Success]) -> float:
    return results.df["success"].mean()
