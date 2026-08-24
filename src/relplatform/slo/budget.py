"""Error budget arithmetic: downtime-in-window, budget consumed/remaining.

Downtime is derived from incident started_at/resolved_at overlap with the measurement
window (clipped to the window boundary for incidents that started before or end after
it) -- see the assumption noted in config/slo.yaml: this treats 100% of an incident's
duration as full downtime for that service, which overstates impact for a degraded-but-
not-fully-down incident. Stated explicitly, not hidden.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from relplatform.slo.config import SLOTarget


def downtime_minutes_in_window(incidents: pd.DataFrame, service: str, window_end: datetime, window_hours: float) -> float:
    """Sum of incident duration for `service` that overlaps [window_end - window_hours,
    window_end], clipped to the window boundary."""
    window_start = window_end - timedelta(hours=window_hours)
    svc = incidents[incidents["service"] == service]
    total_minutes = 0.0
    for row in svc.itertuples():
        clipped_start = max(row.started_at, window_start)
        clipped_end = min(row.resolved_at, window_end)
        if clipped_end > clipped_start:
            total_minutes += (clipped_end - clipped_start).total_seconds() / 60
    return total_minutes


@dataclass
class ErrorBudgetStatus:
    service: str
    window_days: float
    availability_target_pct: float
    window_minutes: float
    error_budget_minutes: float          # total allowed "bad" minutes in the window; 0 when target == 100%
    downtime_minutes: float               # actual incident-derived downtime in the window
    budget_consumed_pct: float | None      # None only when error_budget_minutes == 0 and downtime == 0 (0/0, undefined)
    budget_remaining_minutes: float         # error_budget_minutes - downtime_minutes; NOT clamped, so overspend shows as negative
    exhausted: bool


def compute_error_budget(incidents: pd.DataFrame, target: SLOTarget, as_of: datetime) -> ErrorBudgetStatus:
    window_minutes = target.measurement_window_days * 24 * 60
    error_budget_minutes = window_minutes * (1 - target.availability_target_pct / 100)
    downtime_minutes = downtime_minutes_in_window(incidents, target.service, as_of, target.measurement_window_days * 24)

    if error_budget_minutes <= 0:
        # 100% target: there is no allowed downtime at all, so "percent consumed" of a
        # zero-size budget is undefined at zero downtime, and effectively infinite at any
        # downtime > 0 -- reported as None/inf rather than raising a ZeroDivisionError.
        budget_consumed_pct = 0.0 if downtime_minutes == 0 else float("inf")
        exhausted = downtime_minutes > 0
    else:
        budget_consumed_pct = downtime_minutes / error_budget_minutes * 100
        exhausted = downtime_minutes >= error_budget_minutes

    budget_remaining_minutes = error_budget_minutes - downtime_minutes

    return ErrorBudgetStatus(
        service=target.service,
        window_days=target.measurement_window_days,
        availability_target_pct=target.availability_target_pct,
        window_minutes=window_minutes,
        error_budget_minutes=round(error_budget_minutes, 2),
        downtime_minutes=round(downtime_minutes, 2),
        budget_consumed_pct=None if budget_consumed_pct is None else round(budget_consumed_pct, 2),
        budget_remaining_minutes=round(budget_remaining_minutes, 2),
        exhausted=exhausted,
    )
