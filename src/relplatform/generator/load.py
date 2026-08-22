"""Loads a SimResult into DuckDB."""
from __future__ import annotations

import pandas as pd

from relplatform.generator.graph import CALL_EDGES, SERVICE_META
from relplatform.generator.simulate import SimResult


def load(con, result: SimResult) -> None:
    services_df = pd.DataFrame(
        [{"service": s, "tier": m["tier"], "baseline_risk": m["baseline_risk"]} for s, m in SERVICE_META.items()]
    )
    deps_df = pd.DataFrame([{"upstream": a, "downstream": b} for a, b in CALL_EDGES])
    deployments_df = pd.DataFrame(result.deployments)
    incidents_df = pd.DataFrame(result.incidents)
    alerts_df = pd.DataFrame(result.alerts)
    metrics_df = pd.DataFrame(result.resource_metrics)

    con.execute("DELETE FROM services")
    con.execute("DELETE FROM service_dependencies")
    con.execute("DELETE FROM deployments")
    con.execute("DELETE FROM incidents")
    con.execute("DELETE FROM alerts")
    con.execute("DELETE FROM resource_metrics")

    con.register("services_df", services_df)
    con.execute("INSERT INTO services SELECT * FROM services_df")
    con.register("deps_df", deps_df)
    con.execute("INSERT INTO service_dependencies SELECT * FROM deps_df")
    con.register("deployments_df", deployments_df)
    con.execute("INSERT INTO deployments SELECT * FROM deployments_df")
    con.register("incidents_df", incidents_df)
    con.execute("INSERT INTO incidents SELECT * FROM incidents_df")
    con.register("alerts_df", alerts_df)
    con.execute("INSERT INTO alerts SELECT * FROM alerts_df")
    con.register("metrics_df", metrics_df)
    con.execute("INSERT INTO resource_metrics SELECT * FROM metrics_df")
