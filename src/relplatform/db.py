"""DuckDB connection + schema management."""
from __future__ import annotations

import duckdb

from relplatform.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS services (
    service        VARCHAR PRIMARY KEY,
    tier           VARCHAR,          -- 'edge', 'core', 'data'
    baseline_risk  DOUBLE            -- latent per-service change-failure propensity used by generator
);

CREATE TABLE IF NOT EXISTS service_dependencies (
    upstream    VARCHAR,   -- service that, when it fails, can propagate alerts to `downstream`
    downstream  VARCHAR
);

CREATE TABLE IF NOT EXISTS deployments (
    id             VARCHAR PRIMARY KEY,
    service        VARCHAR,
    deployed_at    TIMESTAMP,
    commit_count   INTEGER,
    lines_changed  INTEGER,
    lead_time_hours DOUBLE,  -- hours from first commit in this batch to deploy (DORA lead time for changes)
    is_weekend     BOOLEAN,
    is_off_hours   BOOLEAN,
    rolled_back    BOOLEAN
);

CREATE TABLE IF NOT EXISTS incidents (
    id                   VARCHAR PRIMARY KEY,
    service              VARCHAR,
    severity             VARCHAR,        -- SEV1..SEV4
    started_at           TIMESTAMP,
    acknowledged_at      TIMESTAMP,
    resolved_at          TIMESTAMP,
    root_cause_category  VARCHAR,
    postmortem_text      VARCHAR,
    triggering_deploy_id VARCHAR,        -- ground truth: which deploy (if any) caused this incident
    origin_service       VARCHAR         -- ground truth: true origin, may differ from `service` (impact) for propagated incidents
);

CREATE TABLE IF NOT EXISTS alerts (
    id           VARCHAR PRIMARY KEY,
    service      VARCHAR,
    severity     VARCHAR,
    message      VARCHAR,
    fired_at     TIMESTAMP,
    resolved_at  TIMESTAMP,
    incident_id  VARCHAR    -- ground truth cluster label; NOT to be used as a clustering feature, eval-only
);

CREATE TABLE IF NOT EXISTS resource_metrics (
    service     VARCHAR,
    ts          TIMESTAMP,
    metric_name VARCHAR,     -- e.g. 'db_connection_pool_utilization_pct'
    value       DOUBLE
);

CREATE TABLE IF NOT EXISTS ai_cache (
    content_hash  VARCHAR PRIMARY KEY,   -- sha256(task || provider || model || input)
    task          VARCHAR,
    provider      VARCHAR,
    model         VARCHAR,
    input_text    VARCHAR,
    output_json   VARCHAR,
    tokens_in     INTEGER,
    tokens_out    INTEGER,
    latency_ms    DOUBLE,
    created_at    TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS alert_clusters (
    run_id        VARCHAR,
    alert_id      VARCHAR,
    cluster_id    VARCHAR,   -- -1 = noise (DBSCAN)
    service       VARCHAR
);

CREATE TABLE IF NOT EXISTS github_cache (
    cache_key     VARCHAR PRIMARY KEY,   -- sha256(request url)
    url           VARCHAR,
    fetched_at    TIMESTAMP,
    response_json VARCHAR
);

CREATE TABLE IF NOT EXISTS on_call_shifts (
    id           VARCHAR PRIMARY KEY,
    engineer     VARCHAR,
    shift_start  TIMESTAMP,
    shift_end    TIMESTAMP,
    is_holiday   BOOLEAN,   -- shift overlaps a company holiday (see generator/roster.py)
    swapped      BOOLEAN    -- holiday coverage traded to the next engineer in rotation
);
"""


def get_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(DB_PATH, read_only=read_only)
    return con


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA)


def reset_schema(con: duckdb.DuckDBPyConnection) -> None:
    for tbl in ["alerts", "incidents", "deployments", "resource_metrics", "service_dependencies", "services", "on_call_shifts"]:
        con.execute(f"DROP TABLE IF EXISTS {tbl}")
    init_schema(con)
