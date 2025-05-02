from typing import Callable, List, Dict
from codeevaluation.metrics.clustering.baseClusterer import BaseClusterer
from codeevaluation.typing.BagOfProperties import (
    BagOfPropertiesFactory,
    BagOfProperties,
)
from codeevaluation.typing.types import id, query, cluster
import polars as pl


def jaccard_ngrams(s1: str, s2: str, n=3) -> float:
    def ngrams(s):
        return set(s[i : i + n] for i in range(len(s) - n + 1)) or {s}

    n1, n2 = ngrams(s1), ngrams(s2)
    return 1 - len(n1 & n2) / len(n1 | n2)


class ExtendableClusterer(BaseClusterer):
    def __init__(
        self,
        existing_clustering: BagOfProperties[
            id, query, cluster
        ] = BagOfPropertiesFactory[id, query, cluster].new(),
        threshold: float = 0.5,
        distance_func: Callable[[str, str], float] = jaccard_ngrams,
    ):
        self._threshold = threshold
        self._distance_func = distance_func
        self.data: BagOfProperties[id, query, cluster] = existing_clustering
        self.cluster_id_counter = (
            existing_clustering.df.select(pl.col("cluster").max()).to_numpy()[0][0]
            if existing_clustering.df.height > 0
            else 0
        )
        self._clusters: Dict[int, List[str]] = {}
        for row in existing_clustering.df.iter_rows(named=True):
            cluster_id = row["cluster"]
            if cluster_id not in self._clusters:
                self._clusters[cluster_id] = []
            self._clusters[cluster_id].append(row["query"])

    def _assign_cluster(self, string: str) -> int:
        for cid, items in self._clusters.items():
            if all(
                self._distance_func(string, other) <= self._threshold for other in items
            ):
                items.append(string)
                return cid
        cid = self.cluster_id_counter
        self._clusters[cid] = [string]
        self.cluster_id_counter += 1
        return cid

    def add_queries(self, queries: BagOfProperties[id, query]) -> None:
        new_rows = []
        for row in queries.df.iter_rows(named=True):
            row["cluster"] = self._assign_cluster(row["query"])
            new_rows.append(row)

        self.data.df = self.data.df.vstack(pl.DataFrame(new_rows))

    @property
    def clusters(self) -> Dict[int, List[str]]:
        return self._clusters
