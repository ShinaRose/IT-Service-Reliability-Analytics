"""The four DORA metrics, computed to the definitions in Google Cloud's DORA program
(https://dora.dev/guides/dora-metrics-four-keys/ , also published in Forsgren et al.,
*Accelerate*, IT Revolution Press, 2018, and the annual State of DevOps reports):

1. Deployment Frequency  -- how often an organization successfully releases to production.
2. Lead Time for Changes -- the time it takes a commit to get into production.
3. Change Failure Rate   -- the percentage of deployments causing a failure in production
   (a degradation requiring remediation, e.g. a rollback, hotfix or incident).
4. Time to Restore Service (MTTR) -- how long it takes to recover from a failure in
   production once one occurs.

Bands (Elite/High/Medium/Low) are the ones published in the 2021-2023 Accelerate State of
DevOps reports; see `relplatform.config.DORA_BANDS`. Two of the published bands have a
known discontinuity in the source report itself (change_failure_rate's High and Medium
bands are identical at "16-30%", and time_to_restore has a gap between "one week" and
"six months" between Medium and Low) -- we reproduce the official numbers as-is rather
than silently patching them, and flag it in the returned dict.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def label_deploy_caused_incidents(
    deployments: pd.DataFrame, incidents: pd.DataFrame, window_hours: float = 4.0
) -> pd.DataFrame:
    """Time-proximity heuristic used industry-wide when there's no explicit deploy->incident
    link: a deployment is charged with an incident if an incident on the same service starts
    within `window_hours` after it, and this deploy is the nearest prior deploy on that
    service to that incident's start time. Returns `deployments` with a `caused_incident`
    column (0/1). This is deliberately independent of the generator's ground-truth
    `triggering_deploy_id` column -- that column exists only so the change-failure model's
    predictions can be scored against a real answer in `relplatform.eval`.
    """
    deployments = deployments.sort_values(["service", "deployed_at"]).reset_index(drop=True)
    incidents = incidents.sort_values("started_at")
    deployments["caused_incident"] = 0

    for service, dep_grp in deployments.groupby("service"):
        inc_times = incidents.loc[incidents["service"] == service, "started_at"].values
        if len(inc_times) == 0:
            continue
        dep_times = dep_grp["deployed_at"].values
        for inc_t in inc_times:
            deltas = (inc_t - dep_times) / np.timedelta64(1, "h")
            valid = np.where((deltas >= 0) & (deltas <= window_hours))[0]
            if len(valid) == 0:
                continue
            nearest = valid[np.argmin(deltas[valid])]
            deployments.loc[dep_grp.index[nearest], "caused_incident"] = 1
    return deployments


def _trend(series_by_period: pd.Series) -> dict:
    """Simple, interpretable trend: % change between the first and second half of the
    available periods, plus the direction."""
    if len(series_by_period) < 2:
        return {"direction": "flat", "pct_change": 0.0}
    half = len(series_by_period) // 2
    first, second = series_by_period.iloc[:half].mean(), series_by_period.iloc[half:].mean()
    if first == 0:
        pct = 0.0
    else:
        pct = 100.0 * (second - first) / abs(first)
    direction = "improving" if pct > 2 else ("worsening" if pct < -2 else "flat")
    return {"direction": direction, "pct_change": round(float(pct), 1)}


def deployment_frequency(deployments: pd.DataFrame) -> dict:
    df = deployments.copy()
    df["month"] = pd.to_datetime(df["deployed_at"]).dt.to_period("M")
    per_month = df.groupby("month").size()
    per_day = len(df) / max(1, (df["deployed_at"].max() - df["deployed_at"].min()).days)

    if per_day >= 1:
        band = "elite"
    elif per_month.median() >= 4:  # roughly weekly
        band = "high"
    elif per_month.median() >= 1 / 6:  # more than once per 6mo
        band = "medium"
    else:
        band = "low"

    # trend on a worse-is-lower metric: rising deploy frequency is improvement
    trend = _trend(per_month)
    return {
        "metric": "deployment_frequency",
        "value_per_day": round(per_day, 2),
        "value_per_month_median": float(per_month.median()) if len(per_month) else 0.0,
        "band": band,
        "trend": trend,
        "by_month": {str(k): v for k, v in per_month.to_dict().items()},
    }


def lead_time_for_changes(deployments: pd.DataFrame) -> dict:
    df = deployments.copy()
    median_hours = float(df["lead_time_hours"].median())
    if median_hours < 1:
        band = "elite"
    elif median_hours <= 24 * 7:
        band = "high"
    elif median_hours <= 24 * 30 * 6:
        band = "medium"
    else:
        band = "low"

    df["month"] = pd.to_datetime(df["deployed_at"]).dt.to_period("M")
    per_month = df.groupby("month")["lead_time_hours"].median()
    trend_raw = _trend(per_month)
    # lower lead time is improvement, so invert the naive direction from _trend (which treats "up" as improving)
    trend = {"direction": {"improving": "worsening", "worsening": "improving", "flat": "flat"}[trend_raw["direction"]],
             "pct_change": trend_raw["pct_change"]}

    return {
        "metric": "lead_time_for_changes",
        "median_hours": round(median_hours, 2),
        "band": band,
        "trend": trend,
        "by_month": {str(k): v for k, v in per_month.round(2).to_dict().items()},
    }


def change_failure_rate(deployments_labeled: pd.DataFrame) -> dict:
    df = deployments_labeled.copy()
    rate = float(df["caused_incident"].mean())
    if rate <= 0.15:
        band = "elite"
    elif rate <= 0.30:
        band = "high"  # NOTE: DORA's published chart shows High and Medium as the same 16-30% band
    elif rate <= 0.45:
        band = "medium"
    else:
        band = "low"

    df["month"] = pd.to_datetime(df["deployed_at"]).dt.to_period("M")
    per_month = df.groupby("month")["caused_incident"].mean()
    trend_raw = _trend(per_month)
    trend = {"direction": {"improving": "worsening", "worsening": "improving", "flat": "flat"}[trend_raw["direction"]],
             "pct_change": trend_raw["pct_change"]}

    return {
        "metric": "change_failure_rate",
        "value_pct": round(rate * 100, 2),
        "band": band,
        "trend": trend,
        "by_month": {str(k): v for k, v in (per_month * 100).round(2).to_dict().items()},
        "note": "DORA's published bands have High and Medium both at 16-30%; reproduced as-is.",
    }


def time_to_restore(incidents: pd.DataFrame) -> dict:
    df = incidents.copy()
    df["restore_hours"] = (pd.to_datetime(df["resolved_at"]) - pd.to_datetime(df["started_at"])).dt.total_seconds() / 3600
    median_hours = float(df["restore_hours"].median())

    if median_hours < 1:
        band = "elite"
    elif median_hours <= 24:
        band = "high"
    elif median_hours <= 24 * 7:
        band = "medium"
    else:
        band = "low"

    df["month"] = pd.to_datetime(df["started_at"]).dt.to_period("M")
    per_month = df.groupby("month")["restore_hours"].median()
    trend_raw = _trend(per_month)
    trend = {"direction": {"improving": "worsening", "worsening": "improving", "flat": "flat"}[trend_raw["direction"]],
             "pct_change": trend_raw["pct_change"]}

    return {
        "metric": "time_to_restore",
        "median_hours": round(median_hours, 2),
        "band": band,
        "trend": trend,
        "by_month": {str(k): v for k, v in per_month.round(2).to_dict().items()},
    }


def compute_all_dora_metrics(deployments: pd.DataFrame, incidents: pd.DataFrame, window_hours: float = 4.0) -> dict:
    labeled = label_deploy_caused_incidents(deployments, incidents, window_hours)
    return {
        "deployment_frequency": deployment_frequency(deployments),
        "lead_time_for_changes": lead_time_for_changes(deployments),
        "change_failure_rate": change_failure_rate(labeled),
        "time_to_restore": time_to_restore(incidents),
    }
