import pandas as pd

from relplatform.analytics.clustering import evaluate_against_ground_truth


def test_purity_and_ari_perfect_clustering():
    df = pd.DataFrame({
        "incident_id": ["INC-1", "INC-1", "INC-2", "INC-2", "INC-2"],
        "cluster_id": ["c0", "c0", "c1", "c1", "c1"],
    })
    result = evaluate_against_ground_truth(df)
    assert result["purity"] == 1.0
    assert result["adjusted_rand_index"] == 1.0


def test_purity_bounded_and_penalizes_mixed_clusters():
    df = pd.DataFrame({
        "incident_id": ["INC-1", "INC-1", "INC-2", "INC-2"],
        "cluster_id": ["c0", "c0", "c0", "c0"],  # everything dumped in one cluster
    })
    result = evaluate_against_ground_truth(df)
    assert 0.0 <= result["purity"] <= 1.0
    assert result["purity"] == 0.5  # majority class in the single cluster is 2 of 4


def test_purity_bounded_with_background_noise():
    df = pd.DataFrame({
        "incident_id": ["INC-1", "INC-1", None, None],
        "cluster_id": ["c0", "c0", "c1", "c2"],
    })
    result = evaluate_against_ground_truth(df)
    assert 0.0 <= result["purity"] <= 1.0
