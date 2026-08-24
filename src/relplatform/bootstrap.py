"""First-run bootstrap: generates data, clusters alerts, and computes the report if the
DuckDB file is empty. Used by the dashboard so a fresh deployment (e.g. Streamlit
Community Cloud, which never receives the git-ignored 145MB reliability.duckdb file --
that's over GitHub's 100MB limit) works without a manual setup step.

Idempotent: checks table contents first, so a warm restart is a no-op.
"""
from __future__ import annotations

from typing import Callable

import duckdb

from relplatform.analytics.clustering import ClusteringConfig, cluster_alerts
from relplatform.config import RANDOM_SEED, SIM_START_MONTHS_AGO
from relplatform.db import init_schema, reset_schema
from relplatform.generator.load import load
from relplatform.generator.simulate import simulate
from relplatform.pipeline import compute_full_report, persist_report

# Caught specifically, not `except Exception`: a bare except here would silently treat a
# locked file, disk I/O error, or corrupted table as "no data yet" -- and ensure_ready()
# reacts to "no data yet" by calling reset_schema(), which drops and regenerates
# everything. A real error deserves to propagate loudly, not to look like an excuse to
# discard a working 145MB database.
_MISSING_TABLE_ERRORS = (duckdb.CatalogException,)


def has_data(con) -> bool:
    init_schema(con)
    # generator.load.load() runs seven separate INSERTs with no wrapping transaction. If the
    # process is killed between them (a real risk on a resource-capped cold start -- the
    # exact scenario this module exists for), checking only `incidents` would call a
    # generation that got as far as incidents but never reached alerts/resource_metrics/
    # on_call_shifts "done", and every later restart would skip regeneration forever.
    # Checking all four doesn't make load() atomic, but it closes most of the window.
    try:
        counts = con.execute(
            "SELECT (SELECT count(*) FROM incidents), (SELECT count(*) FROM alerts), "
            "(SELECT count(*) FROM resource_metrics), (SELECT count(*) FROM on_call_shifts)"
        ).fetchone()
        return all(c > 0 for c in counts)
    except _MISSING_TABLE_ERRORS:
        return False


def has_report(con) -> bool:
    try:
        n = con.execute("SELECT count(*) FROM reports").fetchone()[0]
        return n > 0
    except _MISSING_TABLE_ERRORS:
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
