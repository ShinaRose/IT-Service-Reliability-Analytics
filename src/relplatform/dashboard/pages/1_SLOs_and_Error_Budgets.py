"""SLOs and error budgets. See relplatform/slo/ for the underlying math and the Google
SRE Workbook citations. This page renders it; it computes nothing itself beyond
formatting and the per-service loop.
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
from relplatform.slo.budget import compute_error_budget
from relplatform.slo.burn_rate import evaluate_burn_rate_alerts, project_exhaustion_date
from relplatform.slo.config import load_slo_config
from relplatform.slo.freeze import recommend

st.set_page_config(page_title="SLOs & Error Budgets", layout="wide", initial_sidebar_state="expanded")
theme.inject()


@st.cache_data
def _load_incidents(_con):
    df = _con.execute("SELECT id, service, severity, started_at, resolved_at FROM incidents").df()
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["resolved_at"] = pd.to_datetime(df["resolved_at"])
    return df


@st.cache_data
def _load_latency(_con):
    df = _con.execute("SELECT service, ts, value FROM resource_metrics WHERE metric_name = 'p95_latency_ms'").df()
    df["ts"] = pd.to_datetime(df["ts"])
    return df


con = ensure_connection()
report = load_report(con, force=False)
incidents = _load_incidents(con)
latency = _load_latency(con)
slo_targets = load_slo_config()

# change_failure_rate per service, already computed and persisted in the main report.
# Reused here rather than recomputed, so the cross-check compares against the exact same
# number shown on the Home page's risk ranking, not a second independent calculation.
risk_df = pd.DataFrame(report["risk_scores"]).set_index("service")

with st.sidebar:
    st.markdown(theme.eyebrow_html("SLO controls"), unsafe_allow_html=True)
    st.title("SLOs & Error Budgets")
    st.caption("Per-service targets come from config/slo.yaml. Edit that file to change them.")

st.markdown(
    f'''<div class="hero">
      {theme.eyebrow_html("Phase 1 · Reliability Engineering")}
      <h1 class="hero-title">Error budgets, burn-rate alerting, and ship/freeze calls.</h1>
      <div class="hero-meta">{len(slo_targets)} services with SLO targets · multi-window multi-burn-rate alerting per the Google SRE Workbook</div>
    </div>''',
    unsafe_allow_html=True,
)

theme.assumption_note(
    "Availability is measured as (1 - downtime_minutes / window_minutes), where downtime "
    "comes from incident started_at/resolved_at overlap with the window. This treats the "
    "full incident duration as downtime for that service, which overstates impact for a "
    "degraded-but-not-fully-down incident. See config/slo.yaml and relplatform/slo/budget.py."
)

as_of = pd.to_datetime(latency["ts"]).max() if len(latency) else incidents["resolved_at"].max()
if pd.isna(as_of):
    st.error("No timestamped data available yet. Run the pipeline first.")
    st.stop()

# ---------------- Per-service overview ----------------
with st.container(border=True):
    theme.panel_header("Error budget", "Budget Status", f"As of {as_of.date().isoformat()} · window per service's own config", accent="amber")

    rows = []
    for service, target in slo_targets.items():
        budget = compute_error_budget(incidents, target, as_of)
        consumed = budget.budget_consumed_pct
        rows.append({
            "service": service,
            "availability_target_pct": target.availability_target_pct,
            "window_days": target.measurement_window_days,
            "downtime_minutes": budget.downtime_minutes,
            "error_budget_minutes": budget.error_budget_minutes,
            "budget_consumed_pct": consumed if consumed is not None and consumed != float("inf") else (float("nan") if consumed is None else 999.0),
            "budget_remaining_minutes": budget.budget_remaining_minutes,
            "exhausted": budget.exhausted,
        })
    budget_df = pd.DataFrame(rows).sort_values("budget_consumed_pct", ascending=False)
    st.dataframe(
        budget_df.style.format({
            "availability_target_pct": "{:.2f}%", "downtime_minutes": "{:.1f}", "error_budget_minutes": "{:.1f}",
            "budget_consumed_pct": "{:.1f}%", "budget_remaining_minutes": "{:.1f}",
        }),
        width="stretch", hide_index=True,
    )

# ---------------- Burn-rate alerts ----------------
with st.container(border=True):
    theme.panel_header(
        "Burn rate", "Multi-Window, Multi-Burn-Rate Alerts",
        "Google SRE Workbook “Alerting on SLOs”: fires only when both the long and short window exceed threshold",
        accent="rose",
    )
    burn_rows = []
    for service, target in slo_targets.items():
        for alert in evaluate_burn_rate_alerts(incidents, target, as_of):
            burn_rows.append({
                "service": service, "rule": alert.rule_name,
                "long_window_hours": alert.long_window_hours, "short_window_hours": round(alert.short_window_hours, 2),
                "threshold": alert.threshold,
                "long_window_burn_rate": alert.long_window_burn_rate,
                "short_window_burn_rate": alert.short_window_burn_rate,
                "firing": alert.firing,
            })
    burn_df = pd.DataFrame(burn_rows)
    firing_df = burn_df[burn_df["firing"]]
    if len(firing_df):
        st.error(f"{len(firing_df)} burn-rate alert(s) firing right now.")
    else:
        st.success("No burn-rate alerts firing.")
    st.dataframe(burn_df, width="stretch", hide_index=True)

# ---------------- Exhaustion projection ----------------
with st.container(border=True):
    theme.panel_header(
        "Projection", "Budget Exhaustion Date",
        "Heuristic range from 6h/24h/7d lookback burn rates, not a statistical confidence interval",
        accent="coral",
    )
    proj_rows = []
    for service, target in slo_targets.items():
        budget = compute_error_budget(incidents, target, as_of)
        proj = project_exhaustion_date(incidents, target, as_of, budget.budget_remaining_minutes)
        proj_rows.append({
            "service": service, "status": proj.status,
            "optimistic_date": proj.optimistic_date, "central_date": proj.central_date, "pessimistic_date": proj.pessimistic_date,
        })
    st.dataframe(pd.DataFrame(proj_rows), width="stretch", hide_index=True)

# ---------------- Ship / freeze ----------------
with st.container(border=True):
    theme.panel_header(
        "Recommendation", "Ship / Freeze",
        "Cross-checked against the existing change-failure model: disagreements are surfaced, not hidden",
        accent="violet",
    )
    for service, target in slo_targets.items():
        budget = compute_error_budget(incidents, target, as_of)
        alerts = evaluate_burn_rate_alerts(incidents, target, as_of)
        cfr_pct = float(risk_df.loc[service, "change_failure_rate"] * 100) if service in risk_df.index else None
        rec = recommend(budget, alerts, change_failure_rate_pct=cfr_pct)

        cols = st.columns([1.2, 1, 3])
        cols[0].markdown(f"**{service}**")
        cols[1].markdown(theme.light_pill(rec.light), unsafe_allow_html=True)
        cols[2].caption("; ".join(rec.reasons))
        if rec.disagreement:
            st.warning(f"**{service}**: {rec.disagreement_note}")

# ---------------- Latency (descriptive, not part of the error budget) ----------------
with st.container(border=True):
    theme.panel_header(
        "Latency", "Latency Target Compliance",
        "Descriptive only, not folded into the availability error budget above",
        accent="blue",
    )
    theme.assumption_note(
        "Latency target compliance is reported separately from the availability error "
        "budget, not blended into one number: they're measured against different signals "
        "(incident downtime vs. a p95 latency series) and mixing them into a single "
        "budget would obscure which one is actually driving a problem."
    )
    lat_rows = []
    for service, target in slo_targets.items():
        svc_lat = latency[latency["service"] == service]
        if svc_lat.empty:
            continue
        current = float(svc_lat.sort_values("ts")["value"].iloc[-1])
        window_start = as_of - pd.Timedelta(days=target.measurement_window_days)
        in_window = svc_lat[(svc_lat["ts"] > window_start) & (svc_lat["ts"] <= as_of)]
        breach_days = int((in_window["value"] > target.latency_target_ms).sum())
        lat_rows.append({
            "service": service, "latency_target_ms": target.latency_target_ms,
            "current_p95_ms": round(current, 1), "days_over_target_in_window": breach_days,
            "days_in_window": len(in_window),
        })
    st.dataframe(pd.DataFrame(lat_rows), width="stretch", hide_index=True)
