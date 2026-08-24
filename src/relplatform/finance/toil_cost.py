"""Toil cost = engineer hours on incidents x hourly rate, grouped by service and by
root cause category.

Engineer-hours is an estimate, not a measurement: this dataset has no per-responder
time tracking, so toil hours = responders_by_severity[severity] * (resolved_at -
acknowledged_at), deliberately using acknowledged_at as the start point rather than
started_at -- the gap between an incident starting and someone acknowledging it is
detection lag, not engineer effort.
"""
from __future__ import annotations

import pandas as pd

from relplatform.finance.config import CostConfig


def toil_hours(response_minutes: float, severity: str, cost_config: CostConfig) -> float:
    response_minutes = max(0.0, response_minutes)
    return cost_config.responders(severity) * response_minutes / 60


def toil_cost_eur(response_minutes: float, severity: str, cost_config: CostConfig) -> float:
    return toil_hours(response_minutes, severity, cost_config) * cost_config.loaded_hourly_rate_eur


def toil_costs(incidents: pd.DataFrame, cost_config: CostConfig) -> pd.DataFrame:
    """Returns `incidents` with `response_minutes`, `toil_hours`, `toil_cost_eur` added."""
    df = incidents.copy()
    df["response_minutes"] = (pd.to_datetime(df["resolved_at"]) - pd.to_datetime(df["acknowledged_at"])).dt.total_seconds() / 60
    df["response_minutes"] = df["response_minutes"].clip(lower=0)
    df["toil_hours"] = [toil_hours(row.response_minutes, row.severity, cost_config) for row in df.itertuples()]
    df["toil_cost_eur"] = df["toil_hours"] * cost_config.loaded_hourly_rate_eur
    return df


def toil_by_service(incidents_with_toil: pd.DataFrame) -> pd.DataFrame:
    return (
        incidents_with_toil.groupby("service")
        .agg(toil_hours=("toil_hours", "sum"), toil_cost_eur=("toil_cost_eur", "sum"), n_incidents=("id", "count"))
        .reset_index()
        .sort_values("toil_cost_eur", ascending=False)
    )


def toil_by_root_cause(incidents_with_toil: pd.DataFrame) -> pd.DataFrame:
    return (
        incidents_with_toil.groupby("root_cause_category")
        .agg(toil_hours=("toil_hours", "sum"), toil_cost_eur=("toil_cost_eur", "sum"), n_incidents=("id", "count"))
        .reset_index()
        .sort_values("toil_cost_eur", ascending=False)
    )
