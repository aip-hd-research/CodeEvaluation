from abc import ABC, abstractmethod
from typing import Dict, List
from codeevaluation.typing.BagOfProperties import (
    BagOfPropertiesFactory,
    BagOfProperties,
)
from codeevaluation.typing.types import cluster, short_name, size


class BaseClusterer(ABC):
    """
    Abstract base class for clusterers that defines and implements cluster metrics.
    Subclasses must define self.clusters as: Dict[int, List[str]]
    """

    def get_cluster_metrics(self) -> BagOfProperties[cluster, short_name, size]:
        data = []
        for cluster_id, queries in self.clusters.items():
            name = queries[0][:50] if queries else ""
            data.append(
                {
                    "cluster": cluster_id,
                    "short_name": name,
                    "size": len(queries),
                }
            )
        return BagOfPropertiesFactory[cluster, short_name, size].from_dicts(data)

    @property
    @abstractmethod
    def clusters(self) -> Dict[int, List[str]]:
        """
        Subclasses must implement this property.
        Should return a dictionary mapping cluster ID to list of query strings.
        """
        raise NotImplementedError
