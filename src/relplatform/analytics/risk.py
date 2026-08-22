"""Service risk score: combines incident frequency, MTTR p90, and change failure rate
into a single ranked list of where to spend engineering effort.

Each component is min-max normalized to [0,1] across services (so the score is always
relative to this fleet, not an absolute scale), then combined with configurable
weights. Equal weighting by default -- frequency, severity-of-recovery (p90 MTTR), and
change risk are treated as equally important signals unless a caller has a reason to
weight one higher.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_WEIGHTS = {"incident_frequency": 1 / 3, "mttr_p90": 1 / 3, "change_failure_rate": 1 / 3}


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def compute_risk_scores(
    incidents: pd.DataFrame,
    deployments_labeled: pd.DataFrame,
    mttr_fits: dict,
    months: float,
    weights: dict | None = None,
) -> pd.DataFrame:
    weights = weights or DEFAULT_WEIGHTS
    services = sorted(set(incidents["service"]) | set(deployments_labeled["service"]))

    freq = incidents.groupby("service").size().reindex(services, fill_value=0) / max(months, 1e-6)

    p90 = {}
    for svc in services:
        fit = mttr_fits.get(svc)
        if not fit:
            p90[svc] = 0.0
        elif fit.get("fit"):
            p90[svc] = fit["fitted_percentiles_minutes"]["p90"]
        else:
            p90[svc] = fit["empirical_percentiles"].get("p90", 0.0)
    p90_series = pd.Series(p90)

    cfr = deployments_labeled.groupby("service")["caused_incident"].mean().reindex(services, fill_value=0.0)

    df = pd.DataFrame({
        "service": services,
        "incidents_per_month": freq.values,
        "mttr_p90_minutes": p90_series.reindex(services).values,
        "change_failure_rate": cfr.values,
    })

    df["norm_incident_frequency"] = _minmax(df["incidents_per_month"])
    df["norm_mttr_p90"] = _minmax(df["mttr_p90_minutes"])
    df["norm_change_failure_rate"] = _minmax(df["change_failure_rate"])

    df["risk_score"] = (
        weights["incident_frequency"] * df["norm_incident_frequency"]
        + weights["mttr_p90"] * df["norm_mttr_p90"]
        + weights["change_failure_rate"] * df["norm_change_failure_rate"]
    ) * 100

    return df.sort_values("risk_score", ascending=False).reset_index(drop=True)
