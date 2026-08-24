"""Multi-window, multi-burn-rate SLO alerting, per the Google SRE Workbook's "Alerting on
SLOs" chapter: https://sre.google/workbook/alerting-on-slos/ (see "The Multiwindow,
Multi-Burn-Rate Alert" section in particular).

Burn rate definition (workbook, "Recommended Burn Rate Response"): the rate at which a
service is consuming its error budget relative to the rate that would exactly exhaust it
over the full SLO period. Formally, burn_rate = (observed bad-event rate) / (allowed
bad-event rate) = (1 - SLI) / (1 - SLO). This ratio is period-independent -- a burn rate
of 1 means "consuming budget exactly on pace to exhaust it at the end of the period", a
burn rate of 14.4 means "consuming it 14.4x faster than sustainable".

The two named rules asked for map directly onto the workbook's own worked examples:
  - fast burn: 2% of budget in 1 hour  -> threshold = 0.02 * (period_hours / 1)  = 14.4 at a 30-day period
  - slow burn: 5% of budget in 6 hours -> threshold = 0.05 * (period_hours / 6) = 6.0  at a 30-day period
(14.4 and 6.0 are the exact numbers in the workbook's reference table; we derive them
from the configured measurement_window_days per service rather than hardcoding them, so
a service with a different SLO period still gets a correctly-scaled threshold.)

Multi-window: each rule checks a long window (where the threshold is defined) AND a
short window at 1/12th the length, firing only when BOTH exceed the threshold. This is
the workbook's own mechanism for two things at once: suppressing alerts from a brief
blip that doesn't reflect sustained burn, and clearing an alert quickly once the short
window's rate drops back down even before the long window rolls off.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from relplatform.slo.budget import downtime_minutes_in_window
from relplatform.slo.config import SLOTarget

# (rule_name, budget_fraction, long_window_hours). Short window is long_window_hours/12,
# matching the workbook's 1h/5min and 6h/30min pairings.
BURN_RATE_RULES = [
    ("fast_burn", 0.02, 1.0),
    ("slow_burn", 0.05, 6.0),
]


def _observed_burn_rate(downtime_minutes: float, window_minutes: float, availability_target_pct: float) -> float | None:
    allowed_bad_fraction = 1 - availability_target_pct / 100
    observed_bad_fraction = downtime_minutes / window_minutes if window_minutes > 0 else 0.0
    if allowed_bad_fraction <= 0:
        # 100% target: any downtime at all is an infinite burn rate relative to "allowed" (zero).
        return None if downtime_minutes == 0 else float("inf")
    return observed_bad_fraction / allowed_bad_fraction


@dataclass
class BurnRateAlert:
    service: str
    rule_name: str
    budget_fraction: float
    long_window_hours: float
    short_window_hours: float
    threshold: float
    long_window_burn_rate: float | None
    short_window_burn_rate: float | None
    firing: bool


def evaluate_burn_rate_alerts(incidents: pd.DataFrame, target: SLOTarget, as_of: datetime) -> list[BurnRateAlert]:
    period_hours = target.measurement_window_days * 24
    alerts = []
    for rule_name, budget_fraction, long_hours in BURN_RATE_RULES:
        short_hours = long_hours / 12
        threshold = budget_fraction * (period_hours / long_hours)

        long_downtime = downtime_minutes_in_window(incidents, target.service, as_of, long_hours)
        short_downtime = downtime_minutes_in_window(incidents, target.service, as_of, short_hours)

        long_rate = _observed_burn_rate(long_downtime, long_hours * 60, target.availability_target_pct)
        short_rate = _observed_burn_rate(short_downtime, short_hours * 60, target.availability_target_pct)

        firing = (
            long_rate is not None and short_rate is not None
            and long_rate >= threshold and short_rate >= threshold
        )
        alerts.append(BurnRateAlert(
            service=target.service, rule_name=rule_name, budget_fraction=budget_fraction,
            long_window_hours=long_hours, short_window_hours=short_hours, threshold=round(threshold, 2),
            long_window_burn_rate=None if long_rate is None else round(long_rate, 3) if long_rate != float("inf") else long_rate,
            short_window_burn_rate=None if short_rate is None else round(short_rate, 3) if short_rate != float("inf") else short_rate,
            firing=firing,
        ))
    return alerts


@dataclass
class ExhaustionProjection:
    service: str
    status: str  # "no_recent_burn" | "projected" | "already_exhausted"
    central_date: str | None
    optimistic_date: str | None   # slowest recent burn rate (best case)
    pessimistic_date: str | None  # fastest recent burn rate (worst case)
    note: str


def project_exhaustion_date(incidents: pd.DataFrame, target: SLOTarget, as_of: datetime, remaining_minutes: float) -> ExhaustionProjection:
    """Not a statistical confidence interval -- there's no distributional model here.
    The "interval" is a sensitivity range: burn rate computed over three different
    recent lookback windows (6h, 24h, 7d), giving an optimistic (slowest recent burn)
    and pessimistic (fastest recent burn) exhaustion date around the central 24h-based
    estimate. Stated as a heuristic range, not a confidence interval, deliberately."""
    if remaining_minutes <= 0:
        return ExhaustionProjection(target.service, "already_exhausted", None, None, None,
                                     "Budget is already exhausted for the current window.")

    def _daily_burn_rate_minutes_per_day(lookback_hours: float) -> float:
        downtime = downtime_minutes_in_window(incidents, target.service, as_of, lookback_hours)
        return downtime / (lookback_hours / 24)

    rates = {
        "6h": _daily_burn_rate_minutes_per_day(6),
        "24h": _daily_burn_rate_minutes_per_day(24),
        "7d": _daily_burn_rate_minutes_per_day(24 * 7),
    }
    central_rate = rates["24h"]
    if all(r <= 0 for r in rates.values()):
        return ExhaustionProjection(target.service, "no_recent_burn", None, None, None,
                                     "No incident downtime in the last 7 days -- no exhaustion trend to project.")

    def _date_at_rate(rate: float) -> str | None:
        if rate <= 0:
            return None
        days_to_exhaustion = remaining_minutes / rate
        return (as_of + pd.Timedelta(days=min(days_to_exhaustion, 3650))).date().isoformat()

    slowest_rate = min(r for r in rates.values() if r > 0) if any(r > 0 for r in rates.values()) else 0
    fastest_rate = max(rates.values())

    return ExhaustionProjection(
        service=target.service, status="projected",
        central_date=_date_at_rate(central_rate) if central_rate > 0 else _date_at_rate(fastest_rate),
        optimistic_date=_date_at_rate(slowest_rate),
        pessimistic_date=_date_at_rate(fastest_rate),
        note="Range from burn rate computed over 6h/24h/7d lookback windows, not a statistical confidence interval.",
    )
