"""Capacity forecasting: linear trend projection of a resource metric to a threshold
breach date, with a confidence interval on the slope propagated into a date range.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from scipy import stats


def forecast_service(
    metrics: pd.DataFrame, service: str, metric_name: str = "db_connection_pool_utilization_pct",
    threshold: float = 90.0,
) -> dict:
    df = metrics[(metrics["service"] == service) & (metrics["metric_name"] == metric_name)].sort_values("ts")
    if len(df) < 14:
        return {"service": service, "metric": metric_name, "status": "insufficient_data"}

    t0 = df["ts"].min()
    x = (pd.to_datetime(df["ts"]) - t0).dt.total_seconds() / 86400.0  # days
    y = df["value"].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    current_value = float(y[-1])
    latest_date = pd.to_datetime(df["ts"]).max()

    if slope <= 1e-6:
        return {
            "service": service, "metric": metric_name, "status": "stable_or_declining",
            "current_value": round(current_value, 1), "slope_per_day": round(float(slope), 4),
            "threshold": threshold,
        }

    # A positive slope alone isn't enough to call a breach date "projected". Two gates:
    # p < 0.05 (statistically significant) is not sufficient by itself -- with ~365 daily
    # points, even a practically-flat slope reaches p < 0.05 on sample size alone, which
    # without an effect-size check produced confident-looking dates (sitting exactly at
    # the 10-year cap) for services with r_squared as low as 0.02. Require r_squared >=
    # 0.1 too, so "projected" means the trend actually explains a meaningful share of
    # the variance, not just "technically nonzero and enough days to prove it".
    r_squared = float(r_value ** 2)
    if p_value >= 0.05 or r_squared < 0.10:
        return {
            "service": service, "metric": metric_name, "status": "no_significant_trend",
            "current_value": round(current_value, 1), "slope_per_day": round(float(slope), 4),
            "r_squared": round(r_squared, 3), "p_value": round(float(p_value), 3),
            "threshold": threshold,
        }

    MAX_HORIZON_DAYS = 3650  # cap projections at ~10 years out; beyond that "someday" isn't useful

    def _clamp_days(d: float | None) -> float | None:
        if d is None:
            return None
        return float(np.clip(d, 0, MAX_HORIZON_DAYS))

    days_to_breach = _clamp_days((threshold - current_value) / slope)
    # propagate slope uncertainty (95% CI) into an optimistic/pessimistic breach date
    slope_hi = slope + 1.96 * std_err
    slope_lo = max(1e-6, slope - 1.96 * std_err)
    days_to_breach_optimistic = _clamp_days((threshold - current_value) / slope_hi) if slope_hi > 0 else None
    days_to_breach_pessimistic = _clamp_days((threshold - current_value) / slope_lo) if slope_lo > 0 else None

    projected_date = latest_date + timedelta(days=days_to_breach)
    status = "breach_projected" if days_to_breach > 0 else "already_at_or_above_threshold"

    return {
        "service": service,
        "metric": metric_name,
        "status": status,
        "current_value": round(current_value, 1),
        "threshold": threshold,
        "slope_per_day": round(float(slope), 4),
        "r_squared": round(r_squared, 3),
        "projected_breach_date": projected_date.date().isoformat(),
        "projected_breach_date_optimistic": (latest_date + timedelta(days=days_to_breach_optimistic)).date().isoformat()
            if days_to_breach_optimistic is not None else None,
        "projected_breach_date_pessimistic": (latest_date + timedelta(days=days_to_breach_pessimistic)).date().isoformat()
            if days_to_breach_pessimistic is not None else None,
    }


def forecast_all_services(metrics: pd.DataFrame, metric_name: str = "db_connection_pool_utilization_pct",
                           threshold: float = 90.0) -> list[dict]:
    return [forecast_service(metrics, svc, metric_name, threshold) for svc in sorted(metrics["service"].unique())]
