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

# Numeric band boundaries, factored out as named constants rather than left as inline
# literals in each function's if/elif chain -- relplatform.finance.counterfactual needs
# the exact same thresholds to compute "what if this service moved up one band" without
# silently drifting from what these functions actually implement.
CHANGE_FAILURE_RATE_ELITE_MAX_PCT = 15.0
CHANGE_FAILURE_RATE_HIGH_MAX_PCT = 30.0  # "high" and "medium" are the same 16-30% band in the source report

TIME_TO_RESTORE_ELITE_MAX_HOURS = 1.0
TIME_TO_RESTORE_HIGH_MAX_HOURS = 24.0
TIME_TO_RESTORE_MEDIUM_MAX_HOURS = 24.0 * 7


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
    # Matches config.DORA_BANDS exactly: elite 0-15%, high AND medium both 16-30% (the
    # official chart really does publish the same numeric band twice under two labels --
    # see the module docstring), low is anything above 30%. This used to have a fabricated
    # "medium: 30-45%" band that appears nowhere in the source material, silently
    # mislabeling a >30% rate as medium instead of low.
    if rate <= CHANGE_FAILURE_RATE_ELITE_MAX_PCT / 100:
        band = "elite"
    elif rate <= CHANGE_FAILURE_RATE_HIGH_MAX_PCT / 100:
        band = "high"  # DORA's published chart shows High and Medium as the same 16-30% band
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
    # relplatform.analytics.mttr filters restore_minutes > 0 before fitting/reporting
    # percentiles for the same underlying data; this metric was computing the median over
    # unfiltered restore_hours, so a bad row (resolved_at <= started_at -- clock skew, or
    # an incident marked resolved at creation) silently pulled the DORA median down without
    # mttr_fits agreeing, and could flip the reported band.
    df = df[df["restore_hours"] > 0]
    median_hours = float(df["restore_hours"].median())

    if median_hours < TIME_TO_RESTORE_ELITE_MAX_HOURS:
        band = "elite"
    elif median_hours <= TIME_TO_RESTORE_HIGH_MAX_HOURS:
        band = "high"
    elif median_hours <= TIME_TO_RESTORE_MEDIUM_MAX_HOURS:
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


def compute_all_dora_metrics(
    deployments: pd.DataFrame, incidents: pd.DataFrame, window_hours: float = 4.0,
    labeled: pd.DataFrame | None = None,
) -> dict:
    """`labeled` lets a caller that already ran label_deploy_caused_incidents (e.g.
    pipeline.compute_full_report, which needs it again for the change-failure model and
    risk scoring) pass it in instead of paying for the O(n_incidents x n_deploys)
    time-proximity heuristic a second time on the same inputs."""
    if labeled is None:
        labeled = label_deploy_caused_incidents(deployments, incidents, window_hours)
    return {
        "deployment_frequency": deployment_frequency(deployments),
        "lead_time_for_changes": lead_time_for_changes(deployments),
        "change_failure_rate": change_failure_rate(labeled),
        "time_to_restore": time_to_restore(incidents),
    }
