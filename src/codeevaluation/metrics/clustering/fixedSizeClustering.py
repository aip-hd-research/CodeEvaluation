from typing import Callable, List, Dict, Any
from codeevaluation.metrics.clustering.baseClusterer import BaseClusterer
from codeevaluation.typing.BagOfProperties import (
    BagOfPropertiesFactory,
    BagOfProperties,
)
from codeevaluation.typing.types import id, query, cluster

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import ClusterMixin
from sklearn.cluster import DBSCAN
import polars as pl


class FixedSizeClusterer(BaseClusterer):
    def __init__(
        self,
        queries: BagOfProperties[id, query],
        model: ClusterMixin = DBSCAN(eps=0.5, min_samples=1),
        transformQueries: Callable[
            [List[str]], Any
        ] = lambda s: TfidfVectorizer().fit_transform(s),
    ):
        """
        :param queries: Input data to cluster.
        :param model: A scikit-learn clustering model (must support `fit_predict`).
        :param transformQueries: A function that turns list of strings into numeric features.
        """
        self.data: BagOfProperties[id, query, cluster] = BagOfPropertiesFactory[
            id, query, cluster
        ].new()
        self._clusters: Dict[int, List[str]] = {}

        texts = queries.df["query"].to_list()
        X = transformQueries(texts)
        labels = model.fit_predict(X)

        # Attach labels to original data
        self.data.df = queries.df.with_columns(pl.Series(name="cluster", values=labels))

        # Group by cluster ID
        for row in self.data.df.iter_rows(named=True):
            cluster_id = row["cluster"]
            if cluster_id not in self._clusters:
                self._clusters[cluster_id] = []
            self._clusters[cluster_id].append(row["query"])

    @property
    def clusters(self) -> Dict[int, List[str]]:
        return self._clusters
