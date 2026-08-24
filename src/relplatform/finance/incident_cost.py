"""Incident cost = duration x downtime rate x affected user fraction, per the
spec. Applied per-incident then summed/grouped as needed by callers.
"""
from __future__ import annotations

import pandas as pd

from relplatform.finance.config import CostConfig


def incident_cost_eur(duration_minutes: float, service: str, severity: str, cost_config: CostConfig) -> float:
    duration_minutes = max(0.0, duration_minutes)
    return duration_minutes * cost_config.downtime_rate(service) * cost_config.affected_fraction(severity)


def incident_costs(incidents: pd.DataFrame, cost_config: CostConfig) -> pd.DataFrame:
    """Returns `incidents` with `duration_minutes` and `incident_cost_eur` columns added."""
    df = incidents.copy()
    df["duration_minutes"] = (pd.to_datetime(df["resolved_at"]) - pd.to_datetime(df["started_at"])).dt.total_seconds() / 60
    df["duration_minutes"] = df["duration_minutes"].clip(lower=0)
    df["incident_cost_eur"] = [
        incident_cost_eur(row.duration_minutes, row.service, row.severity, cost_config)
        for row in df.itertuples()
    ]
    return df
