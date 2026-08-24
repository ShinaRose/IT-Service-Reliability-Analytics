"""Financial layer: every reliability metric gets a euro figure. See relplatform/finance/
for the underlying math -- this page renders it and does no computation of its own beyond
per-service loops and formatting.
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

from relplatform.analytics.dora import label_deploy_caused_incidents
from relplatform.dashboard import theme
from relplatform.dashboard.data import ensure_connection, load_report
from relplatform.finance.config import load_cost_config
from relplatform.finance.counterfactual import change_failure_rate_uplift, time_to_restore_uplift
from relplatform.finance.incident_cost import incident_costs
from relplatform.finance.rerank import euro_impact_by_service, side_by_side_ranking
from relplatform.finance.toil_cost import toil_by_root_cause, toil_by_service, toil_costs

st.set_page_config(page_title="Financial Impact", layout="wide", initial_sidebar_state="expanded")
theme.inject()


@st.cache_data
def _load_incidents(_con):
    df = _con.execute(
        "SELECT id, service, severity, started_at, acknowledged_at, resolved_at, root_cause_category FROM incidents"
    ).df()
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["acknowledged_at"] = pd.to_datetime(df["acknowledged_at"])
    df["resolved_at"] = pd.to_datetime(df["resolved_at"])
    return df


@st.cache_data
def _load_deployments(_con):
    df = _con.execute("SELECT id, service, deployed_at FROM deployments").df()
    df["deployed_at"] = pd.to_datetime(df["deployed_at"])
    return df


con = ensure_connection()
report = load_report(con, force=False)
incidents = _load_incidents(con)
deployments = _load_deployments(con)
cost_config = load_cost_config()
risk_df = pd.DataFrame(report["risk_scores"])

with st.sidebar:
    st.markdown(theme.eyebrow_html("Cost controls"), unsafe_allow_html=True)
    st.title("Financial Impact")
    st.caption("Rates come from config/costs.yaml -- edit that file to change them.")
    st.markdown('<div class="control-label">Loaded engineering rate</div>', unsafe_allow_html=True)
    st.caption(f"€{cost_config.loaded_hourly_rate_eur:.0f} / engineer-hour")

st.markdown(
    f'''<div class="hero">
      {theme.eyebrow_html("Phase 2 · Financial Layer")}
      <h1 class="hero-title">Every reliability metric, priced in euros.</h1>
      <div class="hero-meta">{len(incidents)} incidents priced · €{cost_config.loaded_hourly_rate_eur:.0f}/hr loaded engineering rate</div>
    </div>''',
    unsafe_allow_html=True,
)

theme.assumption_note(
    "Downtime cost per service and the loaded engineering rate are illustrative config "
    "values (config/costs.yaml), not measured business figures -- there is no real "
    "finance data behind this platform. Every euro figure on this page inherits that "
    "assumption; edit the config to reflect real numbers."
)

# ---------------- Cost computation ----------------
priced = incident_costs(incidents, cost_config)
priced = toil_costs(priced, cost_config)
total_incident_cost = priced["incident_cost_eur"].sum()
total_toil_cost = priced["toil_cost_eur"].sum()

st.markdown(
    '<div class="stat-grid">'
    + theme.stat_card("Total incident cost", f"€{total_incident_cost:,.0f}", "trailing period, all services")
    + theme.stat_card("Total toil cost", f"€{total_toil_cost:,.0f}", "engineer time on incidents")
    + theme.stat_card("Combined impact", f"€{total_incident_cost + total_toil_cost:,.0f}")
    + theme.stat_card("Incidents priced", f"{len(priced):,}")
    + "</div>",
    unsafe_allow_html=True,
)

# ---------------- Toil by service / root cause ----------------
with st.container(border=True):
    theme.panel_header("Toil", "Toil Cost, by Service and Root Cause",
                        "Engineer hours = responders(severity) x (resolved_at - acknowledged_at)", accent="amber")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("By service")
        st.dataframe(
            toil_by_service(priced).style.format({"toil_hours": "{:.1f}", "toil_cost_eur": "€{:,.0f}"}),
            width="stretch", hide_index=True,
        )
    with col2:
        st.caption("By root cause category")
        st.dataframe(
            toil_by_root_cause(priced).style.format({"toil_hours": "{:.1f}", "toil_cost_eur": "€{:,.0f}"}),
            width="stretch", hide_index=True,
        )

# ---------------- Counterfactual ----------------
with st.container(border=True):
    theme.panel_header("Counterfactual", "If This Service Moved Up One DORA Band",
                        "Modeled for time-to-restore and change-failure-rate only -- see the assumption below", accent="violet")
    theme.assumption_note(
        "Deployment frequency and lead time for changes are NOT modeled here: they correlate "
        "with reliability outcomes across organizations in DORA research, but there's no "
        "non-speculative formula from 'this service deploys more often' to 'this many fewer "
        "incident hours' for one service's own data. Forcing a number out of that correlation "
        "would be exactly the kind of computation this platform doesn't let an LLM do either --"
        " doing it in Python instead wouldn't make it less speculative."
    )

    labeled_deploys = label_deploy_caused_incidents(deployments, incidents)
    observed_days = max(1, (deployments["deployed_at"].max() - deployments["deployed_at"].min()).days)

    cf_rows = []
    for service in sorted(incidents["service"].unique()):
        svc_incidents = priced[priced["service"] == service]

        ttr = time_to_restore_uplift(svc_incidents, service, cost_config)
        cf_rows.append({
            "service": service, "metric": "time_to_restore", "current_band": ttr.current_band,
            "target_band": ttr.target_band, "status": ttr.status,
            "hours_saved_per_year": ttr.hours_saved_per_year, "euros_saved_per_year": ttr.euros_saved_per_year,
            "assumption": ttr.assumption,
        })

        svc_deploys = deployments[deployments["service"] == service]
        deploys_per_year = len(svc_deploys) / observed_days * 365
        svc_labeled = labeled_deploys[labeled_deploys["service"] == service]
        current_rate_pct = float(svc_labeled["caused_incident"].mean() * 100) if len(svc_labeled) else 0.0
        deploy_caused = priced[priced["id"].isin(
            incidents.loc[incidents["service"] == service, "id"]
        )]
        cfr = change_failure_rate_uplift(service, deploys_per_year, current_rate_pct, deploy_caused, cost_config)
        cf_rows.append({
            "service": service, "metric": "change_failure_rate", "current_band": cfr.current_band,
            "target_band": cfr.target_band, "status": cfr.status,
            "hours_saved_per_year": cfr.hours_saved_per_year, "euros_saved_per_year": cfr.euros_saved_per_year,
            "assumption": cfr.assumption,
        })

    cf_df = pd.DataFrame(cf_rows)
    st.dataframe(
        cf_df[["service", "metric", "current_band", "target_band", "status", "hours_saved_per_year", "euros_saved_per_year"]]
        .style.format({"hours_saved_per_year": "{:.0f}", "euros_saved_per_year": "€{:,.0f}"}),
        width="stretch", hide_index=True,
    )
    with st.expander("Show the assumption behind each modeled row"):
        for row in cf_df[cf_df["status"] == "modeled"].itertuples():
            st.caption(f"**{row.service} / {row.metric}**: {row.assumption}")

# ---------------- Risk score vs euro impact ----------------
with st.container(border=True):
    theme.panel_header("Re-rank", "Risk Score vs. Euro Impact",
                        "Same services, two rankings -- rank_delta > 0 means the euro ranking moved it up", accent="coral")
    euro_df = euro_impact_by_service(priced)
    ranking = side_by_side_ranking(risk_df, euro_df)
    st.dataframe(
        ranking[["service", "risk_rank", "euro_rank", "rank_delta", "risk_score", "total_cost_eur"]]
        .style.format({"risk_score": "{:.1f}", "total_cost_eur": "€{:,.0f}"}),
        width="stretch", hide_index=True,
    )
    st.bar_chart(euro_df.set_index("service")["total_cost_eur"], color="#FF9166")
