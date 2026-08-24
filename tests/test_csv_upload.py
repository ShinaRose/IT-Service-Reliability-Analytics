import pandas as pd
import pytest

from relplatform.external.csv_upload import compute_uploaded_dora, map_deployments, map_incidents


# ---------------- map_deployments ----------------

def test_map_deployments_missing_column():
    raw = pd.DataFrame({"date": ["2026-01-01"]})
    result = map_deployments(raw, deployed_at_col="deployed_at")
    assert len(result.df) == 0
    assert any("not found" in e.message for e in result.errors)


def test_map_deployments_no_lead_time_mapping_flags_error():
    raw = pd.DataFrame({"deployed_at": ["2026-01-01T00:00:00Z"]})
    result = map_deployments(raw, deployed_at_col="deployed_at")
    assert result.df.iloc[0]["lead_time_hours"] is pd.NA or pd.isna(result.df.iloc[0]["lead_time_hours"])
    assert any("lead-time or commit-time column" in e.message for e in result.errors)


def test_map_deployments_lead_time_hours_column():
    raw = pd.DataFrame({"deployed_at": ["2026-01-01T00:00:00Z"], "lead_hours": [5.5]})
    result = map_deployments(raw, deployed_at_col="deployed_at", lead_time_hours_col="lead_hours")
    assert result.df.iloc[0]["lead_time_hours"] == pytest.approx(5.5)


def test_map_deployments_derives_lead_time_from_commit_at():
    raw = pd.DataFrame({"deployed_at": ["2026-01-01T12:00:00Z"], "commit_at": ["2026-01-01T00:00:00Z"]})
    result = map_deployments(raw, deployed_at_col="deployed_at", commit_at_col="commit_at")
    assert result.df.iloc[0]["lead_time_hours"] == pytest.approx(12.0)


def test_map_deployments_negative_lead_time_set_to_na_not_clipped():
    raw = pd.DataFrame({"deployed_at": ["2026-01-01T00:00:00Z"], "commit_at": ["2026-01-02T00:00:00Z"]})  # commit AFTER deploy
    result = map_deployments(raw, deployed_at_col="deployed_at", commit_at_col="commit_at")
    assert pd.isna(result.df.iloc[0]["lead_time_hours"])
    assert any("negative lead time" in e.message for e in result.errors)


def test_map_deployments_bad_dates_dropped():
    raw = pd.DataFrame({"deployed_at": ["2026-01-01T00:00:00Z", "not-a-date"]})
    result = map_deployments(raw, deployed_at_col="deployed_at")
    assert len(result.df) == 1
    assert any("could not be parsed" in e.message for e in result.errors)


def test_map_deployments_default_service_when_unmapped():
    raw = pd.DataFrame({"deployed_at": ["2026-01-01T00:00:00Z"]})
    result = map_deployments(raw, deployed_at_col="deployed_at")
    assert result.df.iloc[0]["service"] == "uploaded"


# ---------------- map_incidents ----------------

def test_map_incidents_missing_columns():
    raw = pd.DataFrame({"a": [1]})
    result = map_incidents(raw, started_at_col="started_at", resolved_at_col="resolved_at")
    assert len(result.df) == 0
    assert len(result.errors) == 2


def test_map_incidents_drops_backwards_rows():
    raw = pd.DataFrame({
        "started_at": ["2026-01-01T10:00:00Z", "2026-01-02T10:00:00Z"],
        "resolved_at": ["2026-01-01T12:00:00Z", "2026-01-02T08:00:00Z"],  # second row resolved before started
    })
    result = map_incidents(raw, started_at_col="started_at", resolved_at_col="resolved_at")
    assert len(result.df) == 1
    assert any("before started_at" in e.message for e in result.errors)


def test_map_incidents_bad_dates_dropped():
    raw = pd.DataFrame({"started_at": ["2026-01-01T00:00:00Z", "garbage"], "resolved_at": ["2026-01-01T01:00:00Z", "2026-01-01T02:00:00Z"]})
    result = map_incidents(raw, started_at_col="started_at", resolved_at_col="resolved_at")
    assert len(result.df) == 1


# ---------------- compute_uploaded_dora ----------------

def test_compute_uploaded_dora_empty():
    deployments = map_deployments(pd.DataFrame({"deployed_at": []}), deployed_at_col="deployed_at")
    incidents = map_incidents(pd.DataFrame({"started_at": [], "resolved_at": []}), "started_at", "resolved_at")
    result = compute_uploaded_dora(deployments, incidents)
    assert result.deployment_frequency is None
    assert result.time_to_restore is None
    assert result.n_deployments == 0
    assert result.n_incidents == 0


def test_compute_uploaded_dora_deployments_only():
    raw = pd.DataFrame({
        "deployed_at": pd.date_range("2026-01-01", periods=10, freq="3D"),
        "lead_hours": [2.0] * 10,
    })
    deployments = map_deployments(raw, deployed_at_col="deployed_at", lead_time_hours_col="lead_hours")
    incidents = map_incidents(pd.DataFrame({"started_at": [], "resolved_at": []}), "started_at", "resolved_at")
    result = compute_uploaded_dora(deployments, incidents)
    assert result.deployment_frequency is not None
    assert result.lead_time_for_changes is not None
    assert result.change_failure_rate is not None  # defaults to 0% caused_incident with no incidents
    assert result.time_to_restore is None


def test_compute_uploaded_dora_incidents_only():
    raw = pd.DataFrame({
        "started_at": pd.date_range("2026-01-01", periods=5, freq="10D"),
        "resolved_at": pd.date_range("2026-01-01", periods=5, freq="10D") + pd.Timedelta(hours=2),
    })
    deployments = map_deployments(pd.DataFrame({"deployed_at": []}), deployed_at_col="deployed_at")
    incidents = map_incidents(raw, "started_at", "resolved_at")
    result = compute_uploaded_dora(deployments, incidents)
    assert result.deployment_frequency is None
    assert result.time_to_restore is not None
    assert result.n_incidents == 5


def test_compute_uploaded_dora_links_deploys_to_incidents():
    deploy_times = pd.date_range("2026-01-01", periods=6, freq="10D")
    raw_dep = pd.DataFrame({"deployed_at": deploy_times, "lead_hours": [1.0] * 6})
    # one incident starts 1 hour after the first deploy -> should be flagged caused_incident
    raw_inc = pd.DataFrame({
        "started_at": [deploy_times[0] + pd.Timedelta(hours=1)],
        "resolved_at": [deploy_times[0] + pd.Timedelta(hours=2)],
    })
    deployments = map_deployments(raw_dep, deployed_at_col="deployed_at", lead_time_hours_col="lead_hours")
    incidents = map_incidents(raw_inc, "started_at", "resolved_at")
    result = compute_uploaded_dora(deployments, incidents)
    assert result.change_failure_rate["value_pct"] == pytest.approx(100 / 6, rel=0.01)
