import polars as pl

from codeevaluation.typing.BagOfProperties import (
    BagOfProperties,
    BagOfPropertiesFactory,
)
from codeevaluation.typing.types import id, error, query


def convert_description(error: str) -> str:
    start_index = error.find("Error:")
    if start_index == -1:
        start_index = error.find("Deprecation:")
    if start_index == -1:
        return error.strip()
    end_index = error.find("\\", start_index)
    if end_index == -1:
        end_index = len(error)
    return error[start_index:end_index].strip()


def getQueriesForClustering(
    results: BagOfProperties[id, error],
) -> BagOfProperties[id, query]:
    queries = BagOfPropertiesFactory[id, query].new()

    queries.df = results.df.select(
        [pl.col("id"), pl.col("error").map_elements(convert_description).alias("query")]
    )
    return queries
