"""Re-ranks services by annualized euro impact (incident cost + toil cost) and pairs it
with the existing composite risk score ranking so the difference between "where the
statistical risk model says to look" and "where the money actually is" is visible, not
just each number in isolation.
"""
from __future__ import annotations

import pandas as pd


def euro_impact_by_service(incidents_with_costs: pd.DataFrame) -> pd.DataFrame:
    """incidents_with_costs must already have incident_cost_eur and toil_cost_eur
    columns (relplatform.finance.incident_cost.incident_costs /
    relplatform.finance.toil_cost.toil_costs, both applied)."""
    grouped = (
        incidents_with_costs.groupby("service")
        .agg(
            incident_cost_eur=("incident_cost_eur", "sum"),
            toil_cost_eur=("toil_cost_eur", "sum"),
            n_incidents=("id", "count"),
        )
        .reset_index()
    )
    grouped["total_cost_eur"] = grouped["incident_cost_eur"] + grouped["toil_cost_eur"]
    return grouped.sort_values("total_cost_eur", ascending=False).reset_index(drop=True)


def side_by_side_ranking(risk_df: pd.DataFrame, euro_df: pd.DataFrame) -> pd.DataFrame:
    """risk_df: from relplatform.analytics.risk.compute_risk_scores (or the dashboard's
    live-reweighted equivalent) -- must have `service` and `risk_score`.
    euro_df: from euro_impact_by_service -- must have `service` and `total_cost_eur`.

    Returns one row per service with both ranks and the delta between them, so "moved up
    3 places under the euro ranking" is a direct column, not something a reader has to
    compute by eye from two separate tables.
    """
    risk_ranked = risk_df[["service", "risk_score"]].copy().sort_values("risk_score", ascending=False).reset_index(drop=True)
    risk_ranked["risk_rank"] = risk_ranked.index + 1

    euro_ranked = euro_df[["service", "total_cost_eur"]].copy().sort_values("total_cost_eur", ascending=False).reset_index(drop=True)
    euro_ranked["euro_rank"] = euro_ranked.index + 1

    merged = risk_ranked.merge(euro_ranked, on="service", how="outer")
    merged["rank_delta"] = merged["risk_rank"] - merged["euro_rank"]  # positive = moved UP under euro ranking
    return merged.sort_values("euro_rank").reset_index(drop=True)
