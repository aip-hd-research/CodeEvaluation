# Code Evaluation Library - Demo

# Setup
# Install the conda environment using the setup script as described in CONTRIBUTING.md
# For imporved user experience install the IDE extentions for Black, Flake8 and Pyright.

# For VS Code install the extentions and make the adjustments as described in CONTRIBUTING.md.
# Then execute the tests via the Testing menu of VS Code or with pytest

# Demo

from codeevaluation.typing.BagOfProperties import (
    BagOfProperties,
    BagOfPropertiesFactory,
    SliceBoPType,
)
from codeevaluation.typing.types import (
    d_executable,
    id,
    d_with_params,
    d_translations,
)

from codeevaluation.execution.DExecutor import (
    execute_d_tests,
    fill_d_functions_into_tests,
)

from codeevaluation.metrics.executionResultMetrics import computeSuccessRate
from codeevaluation.metrics.clustering.util import getQueriesForClustering
from codeevaluation.metrics.clustering.fixedSizeClustering import FixedSizeClusterer
from codeevaluation.metrics.clustering.extendableClustering import ExtendableClusterer

from codeevaluation.metrics.clustering.compareClusterings import (
    targetClusterAccuracy,
    specificityScore,
    destructivenessScore,
)

from hydra import initialize, compose
import json
from pathlib import Path

# Typing


def my_function(bagOfProperties: BagOfProperties[id, d_with_params]) -> None:
    ...


def use_my_function(
    bag1: BagOfProperties[id],
    bag2: BagOfProperties[id, d_with_params],
    bag3: BagOfProperties[id, d_with_params, d_translations],
):
    my_function(bag1)
    my_function(bag2)
    my_function(bag3)


# Data Loading
def test_demo():
    Path("demo_results").mkdir(parents=True, exist_ok=True)
    with initialize(config_path="../../conf"):
        cfg = compose(config_name="config")

    dTranslations = BagOfPropertiesFactory[id, d_translations].load_from_huggingface(
        "AIP-Heidelberg/test_code_d_translations"
    )
    dUnitTests = BagOfPropertiesFactory[id, d_with_params].load_from_huggingface(
        "AIP-Heidelberg/test_code_d_with_params"
    )

    # Execution
    dCodeExecutableData = fill_d_functions_into_tests(cfg, dUnitTests, dTranslations)
    dCodeExecutableData.df = dCodeExecutableData.df.head(10)

    @SliceBoPType
    def saveExecutabelData(executableData: BagOfProperties[id, d_executable]):
        executableData.df.write_csv("demo_results/dCodeExecutableData.csv")

    saveExecutabelData(dCodeExecutableData)

    results = execute_d_tests(cfg, dCodeExecutableData)

    results.df.write_csv("demo_results/results.csv")

    # Metrics
    metrics = {}

    metrics["success_rate"] = computeSuccessRate(results)

    queries = getQueriesForClustering(results)
    clusterer = FixedSizeClusterer(queries)

    clusterer.data.df.write_csv("demo_results/clustering.csv")

    clusterer_1_cluster = ExtendableClusterer(threshold=1.0)
    clusterer_1_cluster.add_queries(queries)

    clusterer_1_cluster.data.df.write_csv("demo_results/clustering1.csv")

    metrics["target_cluster_accuracy"] = targetClusterAccuracy(
        old_clustering=clusterer.data,
        new_clustering=clusterer_1_cluster.data,
        targeted_cluster=2,
        success_cluster=0,
    )
    metrics["specificity"] = specificityScore(
        old_clustering=clusterer.data,
        new_clustering=clusterer_1_cluster.data,
        targeted_cluster=2,
        success_cluster=0,
    )
    metrics["destructiveness"] = destructivenessScore(
        old_clustering=clusterer.data,
        new_clustering=clusterer_1_cluster.data,
        success_cluster=0,
    )

    with open("demo_results/metrics.json", "w") as fp:
        json.dump(metrics, fp)
