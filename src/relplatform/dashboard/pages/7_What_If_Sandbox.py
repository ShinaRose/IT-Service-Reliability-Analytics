"""What-if sandbox: MTTR-reduction and change-failure-rate-reduction sliders, re-ranking
both the composite risk score and euro impact live. See relplatform/finance/whatif.py
for the math. Every slider move here is a cheap re-blend of already-computed
per-service signals, not a recompute of anything upstream (no model retraining, no
database round-trip), which is what keeps it interactive.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

try:
    for key in ("RELPLATFORM_PROVIDER", "RELPLATFORM_MONTHS", "GEMINI_API_KEY", "RELPLATFORM_GEMINI_MODEL"):
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass

import pandas as pd

from relplatform.dashboard import theme
from relplatform.dashboard.data import ensure_connection, load_report
from relplatform.finance.config import load_cost_config
from relplatform.finance.incident_cost import incident_costs
from relplatform.finance.rerank import euro_impact_by_service
from relplatform.finance.toil_cost import toil_costs
from relplatform.finance.whatif import whatif_priced_incidents, whatif_risk_scores

st.set_page_config(page_title="What-If Sandbox", layout="wide", initial_sidebar_state="expanded")
theme.inject()

# Same literal thresholds relplatform.analytics.dora.deployment_frequency() uses.
# Reproduced here (not imported) because that function takes a full deployments
# DataFrame with real timestamps to bucket into months, and there's no clean way to
# feed it a single hypothetical "what if the rate were X/day" scalar.
_DEPLOY_FREQ_ELITE_PER_DAY = 1.0
_DEPLOY_FREQ_HIGH_PER_MONTH = 4.0
_DEPLOY_FREQ_MEDIUM_PER_MONTH = 1 / 6


def _deploy_freq_band(per_day: float) -> str:
    per_month = per_day * 30
    if per_day >= _DEPLOY_FREQ_ELITE_PER_DAY:
        return "elite"
    if per_month >= _DEPLOY_FREQ_HIGH_PER_MONTH:
        return "high"
    if per_month >= _DEPLOY_FREQ_MEDIUM_PER_MONTH:
        return "medium"
    return "low"


@st.cache_data
def _load_incidents(_con):
    df = _con.execute(
        "SELECT id, service, severity, started_at, acknowledged_at, resolved_at, root_cause_category FROM incidents"
    ).df()
    for col in ("started_at", "acknowledged_at", "resolved_at"):
        df[col] = pd.to_datetime(df[col])
    return df


@st.cache_data
def _load_deployments(_con):
    df = _con.execute("SELECT deployed_at FROM deployments").df()
    df["deployed_at"] = pd.to_datetime(df["deployed_at"])
    return df


con = ensure_connection()
report = load_report(con, force=False)
incidents = _load_incidents(con)
deployments = _load_deployments(con)
cost_config = load_cost_config()
risk_df = pd.DataFrame(report["risk_scores"])

priced = incident_costs(incidents, cost_config)
priced = toil_costs(priced, cost_config)
baseline_total_eur = priced["incident_cost_eur"].sum() + priced["toil_cost_eur"].sum()

observed_days = max(1, (deployments["deployed_at"].max() - deployments["deployed_at"].min()).days)
current_per_day = len(deployments) / observed_days

with st.sidebar:
    st.markdown(theme.eyebrow_html("What-if controls"), unsafe_allow_html=True)
    st.title("What-If Sandbox")
    st.caption("Every slider here re-blends numbers already on disk. No recompute, no model retraining.")
    mttr_reduction = st.slider("MTTR reduction (%)", 0, 50, 0)
    cfr_reduction = st.slider("Change failure rate reduction (%)", 0, 50, 0)
    st.divider()
    deploy_freq_increase = st.slider("Deploy frequency increase (%, informational only)", 0, 100, 0)

st.markdown(
    f'''<div class="hero">
      {theme.eyebrow_html("Phase 7 · What-If Sandbox")}
      <h1 class="hero-title">Move the sliders, watch the ranking change.</h1>
      <div class="hero-meta">MTTR -{mttr_reduction}% · change failure rate -{cfr_reduction}% · deploy frequency +{deploy_freq_increase}%</div>
    </div>''',
    unsafe_allow_html=True,
)

theme.assumption_note(
    "MTTR reduction compresses every incident's duration by the same ratio (mirrors "
    "finance/counterfactual.py's assumption). Change-failure reduction scales down the "
    "cost of incidents categorized as deployment_regression or configuration_error, "
    "an expected-value adjustment across that category, not a claim about which "
    "specific incident would have been avoided. Deploy frequency has no euro/risk "
    "effect here: there's no non-speculative link from 'deploys more often' to "
    "recovered hours for this data, the same reason finance/counterfactual.py doesn't "
    "model it either."
)

# ---------------- Risk ranking ----------------
with st.container(border=True):
    theme.panel_header("Risk", "Risk Ranking: Before vs. After", "Same three signals, MTTR and change-failure-rate scaled by the sliders", accent="blue")
    whatif_risk = whatif_risk_scores(risk_df, mttr_reduction, cfr_reduction)
    compare = risk_df[["service", "risk_score"]].rename(columns={"risk_score": "risk_score_before"}).merge(
        whatif_risk[["service", "whatif_risk_score"]].rename(columns={"whatif_risk_score": "risk_score_after"}),
        on="service",
    )
    compare["change"] = compare["risk_score_after"] - compare["risk_score_before"]
    compare = compare.sort_values("risk_score_before", ascending=False).reset_index(drop=True)
    st.dataframe(
        compare.style.format({"risk_score_before": "{:.1f}", "risk_score_after": "{:.1f}", "change": "{:+.1f}"}),
        width="stretch", hide_index=True,
    )
    theme.bar_chart(compare.set_index("service")[["risk_score_before", "risk_score_after"]], color=["#5EC8F2", "#4ADE94"])

# ---------------- Euro impact ----------------
with st.container(border=True):
    theme.panel_header("Financial", "Euro Impact: Before vs. After", "Incident + toil cost, same reductions applied", accent="coral")
    whatif_priced = whatif_priced_incidents(priced, mttr_reduction, cfr_reduction)
    whatif_total_eur = whatif_priced["incident_cost_eur"].sum() + whatif_priced["toil_cost_eur"].sum()
    saved_eur = baseline_total_eur - whatif_total_eur

    st.markdown(
        '<div class="stat-grid">'
        + theme.stat_card("Before", f"€{baseline_total_eur:,.0f}")
        + theme.stat_card("After", f"€{whatif_total_eur:,.0f}")
        + theme.stat_card("Saved", f"€{saved_eur:,.0f}", f"{saved_eur / baseline_total_eur * 100:.1f}%" if baseline_total_eur else "")
        + "</div>",
        unsafe_allow_html=True,
    )

    euro_before = euro_impact_by_service(priced)[["service", "total_cost_eur"]].rename(columns={"total_cost_eur": "before"})
    euro_after = euro_impact_by_service(whatif_priced)[["service", "total_cost_eur"]].rename(columns={"total_cost_eur": "after"})
    euro_compare = euro_before.merge(euro_after, on="service").sort_values("before", ascending=False)
    theme.bar_chart(euro_compare.set_index("service")[["before", "after"]], color=["#FF9166", "#4ADE94"])

# ---------------- Combined re-rank ----------------
with st.container(border=True):
    theme.panel_header("Combined", "Risk Rank vs. Euro Rank, After the What-If", "Same side-by-side comparison Phase 2 uses, recomputed against the adjusted numbers", accent="violet")
    risk_ranked = compare[["service", "risk_score_after"]].rename(columns={"risk_score_after": "risk_score"}) \
        .sort_values("risk_score", ascending=False).reset_index(drop=True)
    risk_ranked["risk_rank"] = risk_ranked.index + 1
    euro_ranked = euro_after.rename(columns={"after": "total_cost_eur"}).sort_values("total_cost_eur", ascending=False).reset_index(drop=True)
    euro_ranked["euro_rank"] = euro_ranked.index + 1
    merged = risk_ranked.merge(euro_ranked, on="service")
    merged["rank_delta"] = merged["risk_rank"] - merged["euro_rank"]
    st.dataframe(
        merged[["service", "risk_rank", "euro_rank", "rank_delta", "risk_score", "total_cost_eur"]]
        .sort_values("euro_rank").style.format({"risk_score": "{:.1f}", "total_cost_eur": "€{:,.0f}"}),
        width="stretch", hide_index=True,
    )

# ---------------- Deploy frequency (informational) ----------------
with st.container(border=True):
    theme.panel_header("Deploy Frequency", "Informational Only: No Euro or Risk Effect", "Shown so the slider isn't silently ignored, not because it feeds the numbers above", accent="amber")
    whatif_per_day = current_per_day * (1 + deploy_freq_increase / 100)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current", f"{current_per_day:.2f}/day")
        st.markdown(theme.band_pill(_deploy_freq_band(current_per_day)), unsafe_allow_html=True)
    with col2:
        st.metric("With slider applied", f"{whatif_per_day:.2f}/day")
        st.markdown(theme.band_pill(_deploy_freq_band(whatif_per_day)), unsafe_allow_html=True)
