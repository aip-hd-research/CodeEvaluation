import polars as pl

from codeevaluation.typing.BagOfProperties import (
    BagOfProperties,
)
from codeevaluation.typing.types import id, cluster


def targetClusterAccuracy(
    old_clustering: BagOfProperties[id, cluster],
    new_clustering: BagOfProperties[id, cluster],
    targeted_cluster: int,
    success_cluster: int,
) -> float:
    old_df = old_clustering.df
    new_df = new_clustering.df

    targets = old_df.filter(pl.col("cluster") == targeted_cluster).get_column("id")
    success_cluster_new_df = new_df.filter(pl.col("cluster") == success_cluster)

    targets_in_success_df = success_cluster_new_df.filter(pl.col("id").is_in(targets))
    amount_targets_in_success = targets_in_success_df.height

    return amount_targets_in_success / len(targets) if len(targets) > 0 else 0


def specificityScore(
    old_clustering: BagOfProperties[id, cluster],
    new_clustering: BagOfProperties[id, cluster],
    targeted_cluster: int,
    success_cluster: int,
) -> float:
    # Perc. of samples that did not change cluster (excl. samples from targeted cluster that were corrected)

    old_df = old_clustering.df
    new_df = new_clustering.df
    old_target_df = old_df.filter(pl.col("cluster") == targeted_cluster)

    # Count samples that did not change clusters (excluding targeted cluster)
    non_target_old = old_df.filter(pl.col("cluster") != targeted_cluster)

    merged = non_target_old.join(
        new_df.rename({"cluster": "new_cluster"}), on="id", how="left"
    )
    unchanged = merged.filter(pl.col("cluster") == pl.col("new_cluster")).height

    # Count target samples that changed to non success cluster
    new_target_df = new_df.filter(pl.col("id").is_in(old_target_df.get_column("id")))

    wrongly_changed_target_samples = new_target_df.filter(
        (pl.col("cluster") != targeted_cluster) & (pl.col("cluster") != success_cluster)
    ).height

    total_non_target_samples = non_target_old.height

    return (
        unchanged / (total_non_target_samples + wrongly_changed_target_samples)
        if (total_non_target_samples + wrongly_changed_target_samples) > 0
        else 0
    )


def destructivenessScore(
    old_clustering: BagOfProperties[id, cluster],
    new_clustering: BagOfProperties[id, cluster],
    success_cluster: int,
) -> float:
    old_df = old_clustering.df
    new_df = new_clustering.df

    old_success_ids = (
        old_df.filter(pl.col("cluster") == success_cluster).get_column("id").to_list()
    )
    new_success_ids = (
        new_df.filter(pl.col("cluster") == success_cluster).get_column("id").to_list()
    )

    # Compute which old-success samples are no longer in the new success cluster
    correct_became_faulty = set(old_success_ids) - set(new_success_ids)

    return len(correct_became_faulty) / len(old_success_ids) if old_success_ids else 0
