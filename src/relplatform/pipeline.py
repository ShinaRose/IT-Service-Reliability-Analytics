"""Single source of truth for "compute the full analytics report" -- used by the CLI
pipeline script, the FastAPI service, and the Streamlit dashboard so all three read the
exact same numbers instead of three slightly-different recomputations.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from relplatform.analytics.capacity import forecast_all_services
from relplatform.analytics.change_failure import score_deploys, train_change_failure_model
from relplatform.analytics.clustering import ClusteringConfig, cluster_alerts, evaluate_against_ground_truth, noise_reduction_rate
from relplatform.analytics.dora import compute_all_dora_metrics, label_deploy_caused_incidents
from relplatform.analytics.mttr import fit_mttr_per_service
from relplatform.analytics.risk import compute_risk_scores


def load_tables(con) -> dict:
    return {
        "deployments": con.execute("SELECT * FROM deployments").df(),
        "incidents": con.execute("SELECT * FROM incidents").df(),
        "alerts": con.execute("SELECT id, service, message, fired_at, incident_id FROM alerts").df(),
        "resource_metrics": con.execute("SELECT * FROM resource_metrics").df(),
    }


def compute_full_report(con, use_cached_clusters: bool = True) -> dict:
    tables = load_tables(con)
    deployments, incidents, resource_metrics = tables["deployments"], tables["incidents"], tables["resource_metrics"]

    labeled = label_deploy_caused_incidents(deployments, incidents)
    dora = compute_all_dora_metrics(deployments, incidents, labeled=labeled)
    cf_result = train_change_failure_model(labeled)
    scored_deploys = score_deploys(cf_result["pipeline"], deployments)

    mttr_fits = fit_mttr_per_service(incidents)
    capacity = forecast_all_services(resource_metrics)

    months = max(1.0, (deployments["deployed_at"].max() - deployments["deployed_at"].min()).days / 30.0)
    risk_df = compute_risk_scores(incidents, labeled, mttr_fits, months=months)

    # Full per-deploy risk scores (not just the top 10 in the report JSON) persisted so
    # the dashboard can let a user slide the high-risk flagging percentile and see the
    # result instantly, without retraining the model.
    con.execute("CREATE TABLE IF NOT EXISTS deploy_risk_scores (id VARCHAR, service VARCHAR, deployed_at TIMESTAMP, risk_probability DOUBLE)")
    con.execute("DELETE FROM deploy_risk_scores")
    scores_out = scored_deploys[["id", "service", "deployed_at", "risk_probability"]]
    con.register("scores_out_df", scores_out)
    con.execute("INSERT INTO deploy_risk_scores SELECT * FROM scores_out_df")

    cluster_row = con.execute("SELECT count(*) FROM alert_clusters").fetchone()
    n_cached_clusters = cluster_row[0] if cluster_row else 0
    # Cache is only trusted when its row count matches `alerts` exactly, not just
    # "non-empty": alert_clusters is a full snapshot from one clustering run (see
    # run_clustering.py / bootstrap.py, both `DELETE FROM alert_clusters` before
    # inserting), so a count mismatch means alerts have been added/regenerated since,
    # and the JOIN below is an INNER JOIN -- it would silently drop every alert missing
    # from the stale cache rather than erroring, understating n_alerts with no warning.
    if use_cached_clusters and n_cached_clusters == len(tables["alerts"]) and n_cached_clusters > 0:
        clustered = con.execute(
            "SELECT a.id, a.service, a.message, a.fired_at, a.incident_id, c.cluster_id "
            "FROM alerts a JOIN alert_clusters c ON a.id = c.alert_id"
        ).df()
    else:
        clustered = cluster_alerts(con, tables["alerts"], ClusteringConfig())

    noise = noise_reduction_rate(clustered)
    cluster_eval = evaluate_against_ground_truth(clustered)

    return {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "dora_metrics": dora,
        "change_failure_model": {
            "metrics": {k: v for k, v in cf_result["metrics"].items() if k != "coefficients"},
            "coefficients": cf_result["metrics"]["coefficients"],
            "top_flagged_deploys": scored_deploys.sort_values("risk_probability", ascending=False)
                .head(10)[["id", "service", "risk_probability", "flagged_high_risk"]].to_dict("records"),
        },
        "mttr_fits": mttr_fits,
        "capacity_forecasts": capacity,
        "risk_scores": risk_df.to_dict("records"),
        "noise_reduction": noise,
        "clustering_ground_truth_eval": cluster_eval,
    }


def persist_report(con, report: dict) -> None:
    con.execute(
        "CREATE TABLE IF NOT EXISTS reports (computed_at TIMESTAMP, report_json VARCHAR)"
    )
    con.execute("DELETE FROM reports")
    con.execute("INSERT INTO reports VALUES (?, ?)", [report["computed_at"], json.dumps(report, default=str)])


def load_latest_report(con) -> dict | None:
    row = con.execute("SELECT report_json FROM reports ORDER BY computed_at DESC LIMIT 1").fetchone()
    if row is None:
        return None
    return json.loads(row[0])
