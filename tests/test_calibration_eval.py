import pandas as pd
import pytest

from relplatform.analytics.dora import label_deploy_caused_incidents
from relplatform.eval.calibration_eval import evaluate_change_failure_calibration


def test_evaluate_change_failure_calibration_real_data(sim_result):
    deployments = pd.DataFrame(sim_result.deployments)
    incidents = pd.DataFrame(sim_result.incidents)
    labeled = label_deploy_caused_incidents(deployments, incidents)

    result = evaluate_change_failure_calibration(labeled)
    assert result["n_test"] > 0
    assert 0 <= result["positive_rate_test"] <= 1
    assert result["average_precision"] is not None
    assert result["baseline_average_precision"] == pytest.approx(result["positive_rate_test"])
    # a real, imbalanced dataset should produce a usable calibration curve
    assert result["calibration_curve"] is not None
    assert len(result["calibration_curve"]["predicted_probability"]) >= 2
    assert len(result["calibration_curve"]["predicted_probability"]) == len(result["calibration_curve"]["observed_frequency"])


def test_evaluate_change_failure_calibration_no_positives_in_test_split_skips_curve():
    # 50 deployments, one per day. time_split's default 75/25 chronological cutoff
    # lands at day 38 -- put every positive example in the first 30 days, so the test
    # split (days 38-50) has zero positives, while the full dataset still has enough
    # (8) for 5-fold stratified CV to run without erroring.
    n = 50
    positive_days = {3, 7, 11, 15, 18, 21, 24, 28}
    deployments = pd.DataFrame({
        "id": [f"DEP-{i}" for i in range(n)],
        "service": ["svc"] * n,
        "deployed_at": pd.date_range("2026-01-01", periods=n, freq="1D"),
        "lines_changed": [50] * n,
        "commit_count": [2] * n,
        "lead_time_hours": [1.0] * n,
        "is_weekend": [False] * n,
        "is_off_hours": [False] * n,
        "caused_incident": [1 if (i + 1) in positive_days else 0 for i in range(n)],
    })
    result = evaluate_change_failure_calibration(deployments)
    assert result["positive_rate_test"] == 0.0
    assert result["calibration_curve"] is None
    assert result["average_precision"] is None
