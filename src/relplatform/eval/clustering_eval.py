"""Clustering evaluation: ARI + purity against the generator's ground-truth incident
labels. Not eyeballed -- these are the two standard external cluster-validity metrics
(scikit-learn's adjusted_rand_score, and purity computed directly) run against labels
the clustering algorithm never saw.
"""
from __future__ import annotations

import pandas as pd

from relplatform.analytics.clustering import (
    ClusteringConfig,
    cluster_alerts,
    evaluate_against_ground_truth,
    noise_reduction_rate,
)


def run_clustering_eval(con, alerts: pd.DataFrame, cfg: ClusteringConfig | None = None) -> dict:
    clustered = cluster_alerts(con, alerts, cfg)
    return {
        "noise_reduction": noise_reduction_rate(clustered),
        "ground_truth_eval": evaluate_against_ground_truth(clustered),
        "clustered": clustered,
    }
