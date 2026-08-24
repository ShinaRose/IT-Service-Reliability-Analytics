"""Counterfactual: for each service, if it moved up one DORA band, how many hours and
euros per year would that return.

Only two of the four DORA metrics have a defensible, non-speculative link to recovered
hours/euros from the data this platform has:

  - time_to_restore: moving up a band means a lower MEDIAN restore time. Modeled by
    scaling every incident's duration down by the ratio (target_median / current_median)
    -- i.e. assuming the whole distribution compresses proportionally, not just the
    median shifting while the tail stays put. That's a real, stated assumption, not a
    hidden one: a program that specifically attacks tail incidents (the p99, not the
    median) would recover more than this estimates; a program that only shaves the
    median would recover about this much.

  - change_failure_rate: moving up a band means fewer deploy-caused incidents per year.
    Modeled as (deploys/year) * (current_rate - target_rate) incidents avoided, each
    valued at that service's own historical average incident cost/toil.

deployment_frequency and lead_time_for_changes are NOT modeled here: DORA research shows
they correlate with reliability outcomes across organizations, but there's no
non-speculative formula from "deploys 2x more often" to "this many fewer incident hours"
for one specific service's data the way there is for the other two -- forcing a number
out of that correlation would be exactly the kind of computation this platform's own
rules prohibit an LLM from doing, and doing it in Python instead doesn't make it less
speculative.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from relplatform.analytics.dora import (
    CHANGE_FAILURE_RATE_ELITE_MAX_PCT,
    CHANGE_FAILURE_RATE_HIGH_MAX_PCT,
    TIME_TO_RESTORE_ELITE_MAX_HOURS,
    TIME_TO_RESTORE_HIGH_MAX_HOURS,
    TIME_TO_RESTORE_MEDIUM_MAX_HOURS,
)
from relplatform.finance.config import CostConfig
from relplatform.finance.incident_cost import incident_cost_eur
from relplatform.finance.toil_cost import toil_cost_eur


@dataclass
class BandUplift:
    service: str
    metric: str  # "time_to_restore" | "change_failure_rate"
    current_band: str
    current_value: float
    target_band: str | None
    target_value: float | None
    hours_saved_per_year: float | None
    euros_saved_per_year: float | None
    assumption: str
    status: str  # "modeled" | "already_best_band" | "insufficient_data"


def _time_to_restore_band(median_hours: float) -> tuple[str, float | None]:
    """Returns (current_band, next_better_band_target_hours_or_None)."""
    if median_hours < TIME_TO_RESTORE_ELITE_MAX_HOURS:
        return "elite", None
    if median_hours <= TIME_TO_RESTORE_HIGH_MAX_HOURS:
        return "high", TIME_TO_RESTORE_ELITE_MAX_HOURS
    if median_hours <= TIME_TO_RESTORE_MEDIUM_MAX_HOURS:
        return "medium", TIME_TO_RESTORE_HIGH_MAX_HOURS
    return "low", TIME_TO_RESTORE_MEDIUM_MAX_HOURS


def time_to_restore_uplift(service_incidents: pd.DataFrame, service: str, cost_config: CostConfig) -> BandUplift:
    """service_incidents: this service's incidents over the trailing 12 months, with
    duration_minutes/severity/root_cause_category already present (as produced by
    relplatform.finance.incident_cost.incident_costs)."""
    n = len(service_incidents)
    if n == 0:
        return BandUplift(service, "time_to_restore", "n/a", 0.0, None, None, None, None,
                           "No incidents in the trailing period -- nothing to project.", "insufficient_data")

    current_median_hours = float(service_incidents["duration_minutes"].median() / 60)
    current_band, target_hours = _time_to_restore_band(current_median_hours)

    if target_hours is None:
        return BandUplift(service, "time_to_restore", current_band, round(current_median_hours, 2), None, None,
                           None, None, "Already in the Elite band -- no better band to move up to.", "already_best_band")

    ratio = target_hours / current_median_hours if current_median_hours > 0 else 1.0
    ratio = min(ratio, 1.0)  # never project incidents getting *longer*

    hours_saved = 0.0
    euros_saved = 0.0
    for row in service_incidents.itertuples():
        new_duration_minutes = row.duration_minutes * ratio
        delta_minutes = row.duration_minutes - new_duration_minutes
        hours_saved += delta_minutes / 60
        euros_saved += incident_cost_eur(delta_minutes, service, row.severity, cost_config)
        # toil also falls: same proportional compression applied to response_minutes
        if hasattr(row, "response_minutes"):
            toil_delta_minutes = row.response_minutes * (1 - ratio)
            euros_saved += toil_cost_eur(toil_delta_minutes, row.severity, cost_config)

    band_names = ["low", "medium", "high", "elite"]
    target_band = band_names[band_names.index(current_band) + 1] if current_band in band_names[:-1] else "elite"

    return BandUplift(
        service=service, metric="time_to_restore", current_band=current_band, current_value=round(current_median_hours, 2),
        target_band=target_band, target_value=target_hours,
        hours_saved_per_year=round(hours_saved, 1), euros_saved_per_year=round(euros_saved, 2),
        assumption=(
            f"Assumes every incident's duration compresses by the same ratio "
            f"({target_hours:.1f}h target / {current_median_hours:.1f}h current median = {ratio:.0%}), "
            f"not just the median shifting while the tail stays put."
        ),
        status="modeled",
    )


def _change_failure_rate_band(rate_pct: float) -> tuple[str, float | None]:
    if rate_pct <= CHANGE_FAILURE_RATE_ELITE_MAX_PCT:
        return "elite", None
    if rate_pct <= CHANGE_FAILURE_RATE_HIGH_MAX_PCT:
        return "high", CHANGE_FAILURE_RATE_ELITE_MAX_PCT
    return "low", CHANGE_FAILURE_RATE_HIGH_MAX_PCT


def change_failure_rate_uplift(
    service: str, deploys_per_year: float, current_rate_pct: float,
    deploy_caused_incidents: pd.DataFrame, cost_config: CostConfig,
) -> BandUplift:
    """deploy_caused_incidents: this service's incidents that were flagged as
    deploy-caused (caused_incident==1 from label_deploy_caused_incidents), with cost/toil
    columns already present -- used to price "one avoided incident" for this service."""
    if deploys_per_year <= 0:
        return BandUplift(service, "change_failure_rate", "n/a", current_rate_pct, None, None, None, None,
                           "No deploys in the trailing period -- nothing to project.", "insufficient_data")

    current_band, target_pct = _change_failure_rate_band(current_rate_pct)
    if target_pct is None:
        return BandUplift(service, "change_failure_rate", current_band, round(current_rate_pct, 2), None, None,
                           None, None, "Already in the Elite band -- no better band to move up to.", "already_best_band")

    incidents_avoided_per_year = max(0.0, deploys_per_year * (current_rate_pct - target_pct) / 100)

    if len(deploy_caused_incidents) == 0:
        return BandUplift(service, "change_failure_rate", current_band, round(current_rate_pct, 2),
                           "high" if current_band == "low" else "elite", target_pct, None, None,
                           "No historical deploy-caused incidents for this service to estimate an average cost from.",
                           "insufficient_data")

    avg_incident_hours = float(deploy_caused_incidents["duration_minutes"].mean() / 60)
    avg_incident_euros = float(deploy_caused_incidents["incident_cost_eur"].mean())
    avg_toil_euros = float(deploy_caused_incidents["toil_cost_eur"].mean()) if "toil_cost_eur" in deploy_caused_incidents else 0.0

    hours_saved = incidents_avoided_per_year * avg_incident_hours
    euros_saved = incidents_avoided_per_year * (avg_incident_euros + avg_toil_euros)

    return BandUplift(
        service=service, metric="change_failure_rate", current_band=current_band, current_value=round(current_rate_pct, 2),
        target_band="high" if current_band == "low" else "elite", target_value=target_pct,
        hours_saved_per_year=round(hours_saved, 1), euros_saved_per_year=round(euros_saved, 2),
        assumption=(
            f"Assumes {deploys_per_year:.0f} deploys/year continues unchanged, the rate drops from "
            f"{current_rate_pct:.1f}% to the {target_pct:.0f}% band boundary, and each avoided incident "
            f"costs this service's own historical average ({avg_incident_hours:.1f}h, "
            f"€{avg_incident_euros + avg_toil_euros:,.0f} incl. toil)."
        ),
        status="modeled",
    )
