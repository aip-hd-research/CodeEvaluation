from codeevaluation.typing.BagOfProperties import BagOfPropertiesFactory
from codeevaluation.typing.types import id, error
from codeevaluation.metrics.clustering.extendableClustering import (
    ExtendableClusterer,
    jaccard_ngrams,
)
from codeevaluation.metrics.clustering.fixedSizeClustering import FixedSizeClusterer
from codeevaluation.metrics.clustering.util import getQueriesForClustering
from codeevaluation.metrics.clustering.compareClusterings import (
    targetClusterAccuracy,
    specificityScore,
    destructivenessScore,
)

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


def test_fixedSizeClusterer():
    clusterer = FixedSizeClusterer(queries_bag_0)

    # After adding, we should have as many entries as in the queries_0
    assert clusterer.data.df.height == queries_bag_0.df.height

    # All assigned clusters should be integers
    assert all(isinstance(val, int) for val in clusterer.data.df["cluster"].to_list())


def test_targetClusterAccuracy_on_itself():
    clusterer = FixedSizeClusterer(queries_bag_0)
    accuracy = targetClusterAccuracy(clusterer.data, clusterer.data, 0, 0)

    assert accuracy == 1

    accuracy = targetClusterAccuracy(clusterer.data, clusterer.data, 1, 0)

    assert accuracy == 0


def test_targetClusterAccuracy_on_two_clusterings():
    # Only one cluster
    clusterer1 = ExtendableClusterer(threshold=1.0)
    clusterer1.add_queries(queries_bag_0)

    clusterer2 = FixedSizeClusterer(queries_bag_0)
    accuracy = targetClusterAccuracy(clusterer1.data, clusterer2.data, 0, 0)

    # 345 Successful samples in test queries_bag_0
    assert accuracy == 0.575  # 345 / 600


def test_specificityScore_on_itself():
    clusterer = FixedSizeClusterer(queries_bag_0)
    specificity = specificityScore(clusterer.data, clusterer.data, 5, 0)

    assert specificity == 1


def test_specificityScore_on_two_clusterings():
    # Only one cluster
    clusterer1 = ExtendableClusterer(threshold=0.9)
    clusterer1.add_queries(queries_bag_0)

    clusterer2 = FixedSizeClusterer(queries_bag_0)
    specificity = specificityScore(clusterer1.data, clusterer2.data, 1, 0)

    assert 0 < specificity < 1


def test_destructiveness_on_itself():
    clusterer = FixedSizeClusterer(queries_bag_0)
    destructiveness = destructivenessScore(clusterer.data, clusterer.data, 0)

    assert destructiveness == 0


def test_destructiveness_on_two_clustering():
    # Only one cluster
    clusterer1 = ExtendableClusterer(threshold=1.0)
    clusterer1.add_queries(queries_bag_0)

    clusterer2 = ExtendableClusterer(threshold=0.1)
    clusterer2.add_queries(queries_bag_0)
    destructiveness = destructivenessScore(clusterer1.data, clusterer2.data, 0)

    # 345 Successful samples in test queries_bag_0
    assert destructiveness == 0.425  # 1 - (345 / 600)
