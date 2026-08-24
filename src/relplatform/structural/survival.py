"""Time-between-failures per service via Kaplan-Meier survival analysis: the classic
reliability-engineering "reliability curve" S(t) = P(next failure takes longer than t),
built from inter-arrival gaps between consecutive incidents. The gap from a service's
LAST incident to the end of the observed window is right-censored -- we don't know when
its next failure would have happened yet. Feeding that gap in as an ordinary observed
failure would bias the curve toward "fails sooner than it really does"; Kaplan-Meier is
specifically the tool that handles this correctly instead of silently dropping it or
mistreating it as a real failure.

No external survival-analysis package (e.g. lifelines): the product-limit estimator is a
few lines without the tie-handling complexity a general-purpose library carries (each gap
here is a real-valued minute count with no exact duplicate ties), so an in-house
implementation keeps one fewer dependency in a memory-capped deployment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SurvivalCurve:
    service: str
    n_events: int                          # observed (uncensored) inter-arrival gaps
    n_censored: int                        # 0 or 1: the trailing gap to end-of-window, if any incidents exist
    mtbf_minutes: float | None             # mean of OBSERVED gaps only (censored gap excluded)
    median_survival_minutes: float | None  # first t where S(t) <= 0.5, if the curve reaches it
    timeline: list[float]
    survival: list[float]


def time_between_failures(incidents: pd.DataFrame, service: str, observation_end) -> tuple[np.ndarray, np.ndarray]:
    """Returns (durations_minutes, event_observed) for one service: inter-arrival gaps
    between consecutive incident started_at times (event_observed=True), plus the
    trailing gap from the last incident to `observation_end` marked censored
    (event_observed=False)."""
    starts = pd.to_datetime(incidents.loc[incidents["service"] == service, "started_at"]).sort_values()
    if len(starts) == 0:
        return np.array([]), np.array([], dtype=bool)

    gaps = starts.diff().dropna().dt.total_seconds().to_numpy() / 60
    events = np.ones(len(gaps), dtype=bool)

    tail_gap = (pd.Timestamp(observation_end) - starts.iloc[-1]).total_seconds() / 60
    if tail_gap > 0:
        gaps = np.append(gaps, tail_gap)
        events = np.append(events, False)

    return gaps, events


def kaplan_meier(durations: np.ndarray, event_observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standard product-limit estimator. Returns (timeline, survival_prob), starting at
    (0, 1.0). Only times with at least one observed event create a step down; censored
    observations still shrink the at-risk set for later times, just without a step of
    their own."""
    if len(durations) == 0:
        return np.array([0.0]), np.array([1.0])

    unique_event_times = np.unique(durations[event_observed])
    timeline = [0.0]
    survival = [1.0]
    s = 1.0

    for t in unique_event_times:
        d = int(np.sum((durations == t) & event_observed))  # events exactly at t
        n_at_risk = int(np.sum(durations >= t))              # still under observation just before t
        if n_at_risk > 0:
            s *= (1 - d / n_at_risk)
        timeline.append(float(t))
        survival.append(s)

    return np.array(timeline), np.array(survival)


def reliability_curve(incidents: pd.DataFrame, service: str, observation_end) -> SurvivalCurve:
    durations, events = time_between_failures(incidents, service, observation_end)
    timeline, survival = kaplan_meier(durations, events)

    n_events = int(events.sum())
    observed_gaps = durations[events] if len(durations) else durations
    mtbf = float(np.mean(observed_gaps)) if len(observed_gaps) else None

    median = None
    for t, s in zip(timeline, survival):
        if s <= 0.5:
            median = float(t)
            break

    return SurvivalCurve(
        service=service, n_events=n_events, n_censored=int(len(events) - n_events),
        mtbf_minutes=mtbf, median_survival_minutes=median,
        timeline=[float(t) for t in timeline], survival=[float(s) for s in survival],
    )


def reliability_curves_all_services(incidents: pd.DataFrame, observation_end) -> dict[str, SurvivalCurve]:
    if len(incidents) == 0:
        return {}
    return {
        service: reliability_curve(incidents, service, observation_end)
        for service in sorted(incidents["service"].unique())
    }
