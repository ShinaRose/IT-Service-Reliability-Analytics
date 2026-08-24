"""What-if sandbox: continuous MTTR-reduction and change-failure-rate-reduction
sliders, re-ranking both the composite risk score and the euro impact live.
Deliberately NOT tied to DORA band boundaries the way finance/counterfactual.py is --
that module answers "what does the next official band return," this answers "what if
we improved by exactly X%," so the slider can land anywhere.

A deploy-frequency slider exists on the dashboard page (the spec asks for one), but it
does NOT feed into the euro/risk numbers here, for the same reason
finance/counterfactual.py doesn't model it there either: there's no non-speculative
link from "deploys N% more often" to recovered hours/euros for this data. Forcing one
in for the sake of a third slider having an effect would be exactly the kind of
fabricated precision this platform's rules prohibit.

Fast by construction: every input here is already computed (priced incidents, the risk
dataframe's raw per-service signals) -- a slider move is a handful of pandas
multiplications and a re-normalize, not a recompute of anything upstream (no model
retraining, no database round-trip).
"""
from __future__ import annotations

import pandas as pd

from relplatform.analytics.risk import DEFAULT_WEIGHTS, minmax_normalize

# The two root-cause categories the generator attributes to a deployment (see
# generator/simulate.py: a deploy-triggered incident is "deployment_regression" 78% of
# the time, "configuration_error" otherwise). Used here as the practical,
# incident-level proxy for "this incident was deploy-caused" -- an incident-by-incident
# causal trace isn't available (label_deploy_caused_incidents flags DEPLOYS, not
# incidents), so this is a category-level approximation, stated plainly rather than
# silently assumed exact.
DEPLOY_CAUSED_CATEGORIES = ["deployment_regression", "configuration_error"]


def whatif_priced_incidents(priced_incidents: pd.DataFrame, mttr_reduction_pct: float, cfr_reduction_pct: float) -> pd.DataFrame:
    """Returns `priced_incidents` (already run through finance.incident_cost /
    finance.toil_cost) with incident_cost_eur/toil_cost_eur scaled down:
    - MTTR reduction compresses every incident's cost by the same ratio (mirrors
      finance.counterfactual's proportional-compression assumption).
    - Change-failure reduction scales down only the cost of incidents in
      DEPLOY_CAUSED_CATEGORIES, as an expected-value adjustment across the whole
      category -- not a claim about which specific incident would have been avoided.
    """
    df = priced_incidents.copy()
    mttr_ratio = max(0.0, 1 - mttr_reduction_pct / 100)
    df["incident_cost_eur"] = df["incident_cost_eur"] * mttr_ratio
    if "toil_cost_eur" in df.columns:
        df["toil_cost_eur"] = df["toil_cost_eur"] * mttr_ratio

    if cfr_reduction_pct > 0 and "root_cause_category" in df.columns:
        cfr_ratio = max(0.0, 1 - cfr_reduction_pct / 100)
        is_deploy_caused = df["root_cause_category"].isin(DEPLOY_CAUSED_CATEGORIES)
        df.loc[is_deploy_caused, "incident_cost_eur"] = df.loc[is_deploy_caused, "incident_cost_eur"] * cfr_ratio
        if "toil_cost_eur" in df.columns:
            df.loc[is_deploy_caused, "toil_cost_eur"] = df.loc[is_deploy_caused, "toil_cost_eur"] * cfr_ratio

    return df


def whatif_risk_scores(risk_df: pd.DataFrame, mttr_reduction_pct: float, cfr_reduction_pct: float, weights: dict | None = None) -> pd.DataFrame:
    """Scales the same raw mttr_p90_minutes / change_failure_rate signals
    compute_risk_scores already produced, re-normalizes across services, and
    re-blends -- the same "reweight instantly, no recompute" trick the Home page
    already uses for its weight sliders, extended to also scale the underlying signal
    rather than just its weight."""
    weights = weights or DEFAULT_WEIGHTS
    if len(risk_df) == 0:
        return risk_df.assign(whatif_risk_score=pd.Series(dtype=float))

    df = risk_df.copy()
    mttr_ratio = max(0.0, 1 - mttr_reduction_pct / 100)
    cfr_ratio = max(0.0, 1 - cfr_reduction_pct / 100)

    df["whatif_mttr_p90_minutes"] = df["mttr_p90_minutes"] * mttr_ratio
    df["whatif_change_failure_rate"] = df["change_failure_rate"] * cfr_ratio

    df["norm_incident_frequency"] = minmax_normalize(df["incidents_per_month"])
    df["norm_mttr_p90"] = minmax_normalize(df["whatif_mttr_p90_minutes"])
    df["norm_change_failure_rate"] = minmax_normalize(df["whatif_change_failure_rate"])

    df["whatif_risk_score"] = (
        weights["incident_frequency"] * df["norm_incident_frequency"]
        + weights["mttr_p90"] * df["norm_mttr_p90"]
        + weights["change_failure_rate"] * df["norm_change_failure_rate"]
    ) * 100

    return df.sort_values("whatif_risk_score", ascending=False).reset_index(drop=True)
