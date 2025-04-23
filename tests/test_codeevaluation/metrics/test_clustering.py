from codeevaluation.typing.BagOfProperties import BagOfPropertiesFactory
from codeevaluation.typing.types import id, error
from codeevaluation.metrics.clustering.extendableClustering import (
    ExtendableClusterer,
    jaccard_ngrams,
)
from codeevaluation.metrics.clustering.util import getQueriesForClustering

# Load the data once
results_bag_0 = BagOfPropertiesFactory[id, error].from_json(
    "tests/resources/clustering/results_0.json"
)
results_bag_1 = BagOfPropertiesFactory[id, error].from_json(
    "tests/resources/clustering/results_0.json"
)

queries_bag_0 = getQueriesForClustering(results_bag_0)
queries_bag_1 = getQueriesForClustering(results_bag_1)


def test_empty_initialization():
    clusterer = ExtendableClusterer()

    # Should start with no data
    assert clusterer.data.df.height == 0
    assert clusterer.cluster_id_counter == 0
    assert clusterer.clusters == {}


def test_add_first_queries():
    clusterer = ExtendableClusterer()

    clusterer.add_queries(queries_bag_0)

    # After adding, we should have as many entries as in the queries_0
    assert clusterer.data.df.height == queries_bag_0.df.height

    # All assigned clusters should be integers
    assert all(isinstance(val, int) for val in clusterer.data.df["cluster"].to_list())


def test_add_second_queries():
    clusterer = ExtendableClusterer()

    # First add queries_0
    clusterer.add_queries(queries_bag_0)
    first_count = clusterer.data.df.height

    # Then add queries_1
    clusterer.add_queries(queries_bag_1)
    second_count = clusterer.data.df.height

    assert second_count == first_count + queries_bag_1.df.height

    # Check that newly added queries have clusters assigned
    new_rows = clusterer.data.df.tail(queries_bag_1.df.height)
    assert "cluster" in new_rows.columns


def test_distance_function_basic():
    s1 = "hello world"
    s2 = "hello there"

    distance = jaccard_ngrams(s1, s2)
    assert 0 <= distance <= 1


def test_threshold_effect():
    clusterer_low_threshold = ExtendableClusterer(threshold=0.1)
    clusterer_high_threshold = ExtendableClusterer(threshold=0.9)

    clusterer_low_threshold.add_queries(queries_bag_0)
    clusterer_high_threshold.add_queries(queries_bag_0)

    num_clusters_low = len(clusterer_low_threshold.clusters)
    num_clusters_high = len(clusterer_high_threshold.clusters)

    # With a high threshold, there should generally be fewer clusters
    assert num_clusters_high <= num_clusters_low
