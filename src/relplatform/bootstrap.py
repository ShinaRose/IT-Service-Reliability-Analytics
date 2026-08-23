"""First-run bootstrap: generates data, clusters alerts, and computes the report if the
DuckDB file is empty. Used by the dashboard so a fresh deployment (e.g. Streamlit
Community Cloud, which never receives the git-ignored 145MB reliability.duckdb file --
that's over GitHub's 100MB limit) works without a manual setup step.

Idempotent: checks table contents first, so a warm restart is a no-op.
"""
from __future__ import annotations

from typing import Callable

from relplatform.analytics.clustering import ClusteringConfig, cluster_alerts
from relplatform.config import RANDOM_SEED, SIM_START_MONTHS_AGO
from relplatform.db import init_schema, reset_schema
from relplatform.generator.load import load
from relplatform.generator.simulate import simulate
from relplatform.pipeline import compute_full_report, persist_report


def has_data(con) -> bool:
    init_schema(con)
    try:
        n = con.execute("SELECT count(*) FROM incidents").fetchone()[0]
        return n > 0
    except Exception:
        return False


def has_report(con) -> bool:
    try:
        n = con.execute("SELECT count(*) FROM reports").fetchone()[0]
        return n > 0
    except Exception:
        return False


def ensure_ready(con, progress: Callable[[str], None] = lambda msg: None) -> None:
    """Generates + clusters + computes the report if any step hasn't run yet. Safe to
    call on every startup -- each step is skipped if its output already exists."""
    if not has_data(con):
        progress(f"Generating {SIM_START_MONTHS_AGO} months of synthetic data...")
        result = simulate(seed=RANDOM_SEED, months=SIM_START_MONTHS_AGO)
        reset_schema(con)
        load(con, result)

    cluster_row = con.execute("SELECT count(*) FROM alert_clusters").fetchone()
    if not cluster_row or cluster_row[0] == 0:
        progress("Clustering alerts (downloads the MiniLM embedding model on first run)...")
        alerts = con.execute("SELECT id, service, message, fired_at, incident_id FROM alerts").df()
        clustered = cluster_alerts(con, alerts, ClusteringConfig())
        out = clustered[["id", "cluster_id", "service"]].rename(columns={"id": "alert_id"})
        out.insert(0, "run_id", "bootstrap")
        con.execute("DELETE FROM alert_clusters")
        con.register("out_df", out)
        con.execute("INSERT INTO alert_clusters SELECT * FROM out_df")

    if not has_report(con):
        progress("Computing DORA metrics, risk scores, and forecasts...")
        report = compute_full_report(con)
        persist_report(con, report)
