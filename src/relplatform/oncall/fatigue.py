"""Alert fatigue score per service: combines the existing alert-clustering noise ratio
(how many raw alerts it takes to represent one distinct thing worth looking at) with
paging load (how often that service actually wakes someone up), per the spec's
"alert fatigue score per service combined with existing noise-reduction metric".

Both components are min-max normalized across services and averaged 50/50 -- an
explicit, stated weighting (surfaced in the UI via assumption_note), not a hidden
judgment call. With only 8 services, min-max normalization keeps the score legible
(worst service = 100, best = 0) rather than a z-score that's harder to read at a glance.
"""
from __future__ import annotations

import pandas as pd

NOISE_WEIGHT = 0.5
PAGE_LOAD_WEIGHT = 0.5


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def noise_ratio_by_service(clustered_alerts: pd.DataFrame) -> pd.DataFrame:
    """clustered_alerts: alerts with `service` and `cluster_id` (from
    relplatform.analytics.clustering.cluster_alerts). Returns raw alert count, distinct
    cluster count, and their ratio (>=1; higher = noisier) per service."""
    if len(clustered_alerts) == 0:
        return pd.DataFrame(columns=["service", "raw_alerts", "distinct_clusters", "noise_ratio"])
    grouped = clustered_alerts.groupby("service").agg(
        raw_alerts=("cluster_id", "size"),
        distinct_clusters=("cluster_id", "nunique"),
    ).reset_index()
    grouped["noise_ratio"] = grouped["raw_alerts"] / grouped["distinct_clusters"]
    return grouped


def alert_fatigue_by_service(clustered_alerts: pd.DataFrame, paged_incidents: pd.DataFrame, months: float) -> pd.DataFrame:
    """Returns one row per service with noise_ratio, pages_per_month, and a 0-100
    fatigue_score. `months` is the observed period length, used to annualize page counts
    onto a comparable per-month basis regardless of how much history is loaded."""
    noise = noise_ratio_by_service(clustered_alerts)
    if len(noise) == 0:
        return pd.DataFrame(columns=["service", "raw_alerts", "distinct_clusters", "noise_ratio", "n_pages", "pages_per_month", "fatigue_score"])

    months = max(months, 1e-6)
    pages_by_service = (
        paged_incidents.dropna(subset=["engineer"]).groupby("service").size()
        if len(paged_incidents) else pd.Series(dtype=int)
    )
    noise["n_pages"] = noise["service"].map(pages_by_service).fillna(0).astype(int)
    noise["pages_per_month"] = noise["n_pages"] / months

    noise["fatigue_score"] = 100 * (
        NOISE_WEIGHT * _minmax(noise["noise_ratio"]) + PAGE_LOAD_WEIGHT * _minmax(noise["pages_per_month"])
    )
    return noise.sort_values("fatigue_score", ascending=False).reset_index(drop=True)
