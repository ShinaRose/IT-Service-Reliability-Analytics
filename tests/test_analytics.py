import pandas as pd

from relplatform.analytics.capacity import forecast_all_services
from relplatform.analytics.change_failure import train_change_failure_model
from relplatform.analytics.dora import label_deploy_caused_incidents
from relplatform.analytics.mttr import fit_mttr_per_service
from relplatform.analytics.risk import compute_risk_scores


def test_change_failure_model_trains_and_scores(sim_result):
    deployments = pd.DataFrame(sim_result.deployments)
    incidents = pd.DataFrame(sim_result.incidents)
    labeled = label_deploy_caused_incidents(deployments, incidents)
    result = train_change_failure_model(labeled)
    assert result["metrics"]["n_train"] > 0
    assert result["metrics"]["n_test"] > 0
    assert 0 <= result["test_scores"]["risk_probability"].min()
    assert result["test_scores"]["risk_probability"].max() <= 1


def test_mttr_fit_reports_percentiles(sim_result):
    incidents = pd.DataFrame(sim_result.incidents)
    fits = fit_mttr_per_service(incidents, min_n=3)
    assert len(fits) > 0
    for svc, fit in fits.items():
        assert "p50" in fit.get("fitted_percentiles_minutes", fit.get("empirical_percentiles", {}))


def test_capacity_forecast(sim_result):
    metrics = pd.DataFrame(sim_result.resource_metrics)
    forecasts = forecast_all_services(metrics)
    assert len(forecasts) == 8
    statuses = {f["status"] for f in forecasts}
    assert statuses  # at least runs without error


def test_risk_scores_ranked(sim_result):
    deployments = pd.DataFrame(sim_result.deployments)
    incidents = pd.DataFrame(sim_result.incidents)
    labeled = label_deploy_caused_incidents(deployments, incidents)
    mttr_fits = fit_mttr_per_service(incidents, min_n=3)
    df = compute_risk_scores(incidents, labeled, mttr_fits, months=3)
    assert list(df["risk_score"]) == sorted(df["risk_score"], reverse=True)
    assert set(df["service"]) == set(incidents["service"]) | set(deployments["service"])
