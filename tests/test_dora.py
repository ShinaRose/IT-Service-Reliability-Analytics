import pandas as pd

from relplatform.analytics.dora import compute_all_dora_metrics, label_deploy_caused_incidents


def test_label_deploy_caused_incidents_matches_ground_truth_reasonably(sim_result):
    deployments = pd.DataFrame(sim_result.deployments)
    incidents = pd.DataFrame(sim_result.incidents)
    labeled = label_deploy_caused_incidents(deployments, incidents, window_hours=4.0)

    truth = set(incidents.loc[incidents["triggering_deploy_id"].notna(), "triggering_deploy_id"])
    predicted = set(labeled.loc[labeled["caused_incident"] == 1, "id"])
    overlap = len(truth & predicted)
    # heuristic should recover most true deploy-caused incidents
    assert overlap / max(1, len(truth)) > 0.6


def test_dora_metrics_shape_and_bands(sim_result):
    deployments = pd.DataFrame(sim_result.deployments)
    incidents = pd.DataFrame(sim_result.incidents)
    metrics = compute_all_dora_metrics(deployments, incidents)

    for key in ["deployment_frequency", "lead_time_for_changes", "change_failure_rate", "time_to_restore"]:
        assert key in metrics
        assert metrics[key]["band"] in ("elite", "high", "medium", "low")
        assert "trend" in metrics[key]
