from decimal import Decimal

from codeevaluation.typing.BagOfProperties import BoP
from codeevaluation.typing.column_types import ID, Success


def computeSuccessRate(results: BoP[ID, Success]) -> float:
    mean_value = results.df["success"].mean()
    # Handle various possible types and cast to float
    if isinstance(mean_value, (int, float, Decimal)):
        return float(mean_value)
    raise TypeError(
        f"Invalid type for mean value: {type(mean_value)}. Expected int, float, or Decimal."
    )
