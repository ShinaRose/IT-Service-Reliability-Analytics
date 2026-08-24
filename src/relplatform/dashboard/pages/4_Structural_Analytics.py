"""Structural analytics: dependency-graph blast radius/criticality, mined
failure-propagation edges, change-point early-warning backtest, and time-between-
failures reliability curves. See relplatform/structural/ for the underlying math -- this
page renders it and does no computation of its own beyond formatting and chart layout.
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
from relplatform.dashboard.data import ensure_connection
from relplatform.generator.graph import build_graph
from relplatform.structural.changepoint import ChangePointConfig, backtest_all_services
from relplatform.structural.graph import structural_report
from relplatform.structural.propagation import enrichment_scores, validate_against_dependency_graph
from relplatform.structural.survival import reliability_curves_all_services

st.set_page_config(page_title="Structural Analytics", layout="wide", initial_sidebar_state="expanded")
theme.inject()


@st.cache_data
def _load_incidents(_con):
    df = _con.execute("SELECT id, service, started_at FROM incidents").df()
    df["started_at"] = pd.to_datetime(df["started_at"])
    return df


@st.cache_data
def _load_alerts(_con):
    df = _con.execute("SELECT id, service, fired_at FROM alerts").df()
    df["fired_at"] = pd.to_datetime(df["fired_at"])
    return df


@st.cache_data
def _run_changepoint_backtest(_con, incidents: pd.DataFrame, alerts: pd.DataFrame):
    return backtest_all_services(alerts, incidents, ChangePointConfig())


@st.cache_data
def _run_propagation_mining(_con, incidents: pd.DataFrame):
    return enrichment_scores(incidents, window_minutes=60.0, min_observed=3)


def _combine_curves_for_chart(curves: dict) -> pd.DataFrame:
    """Step-function alignment: each service's KM curve has its own irregular time
    points, so build a shared time grid and forward-fill each curve onto it (a survival
    curve is constant between its own drop points -- this is not an interpolation
    choice, it's what the step function already means)."""
    all_times = sorted({t for c in curves.values() for t in c.timeline})
    data = {}
    for service, c in curves.items():
        s = pd.Series(c.survival, index=c.timeline)
        data[service] = s.reindex(all_times, method="ffill")
    return pd.DataFrame(data, index=all_times)


con = ensure_connection()
incidents = _load_incidents(con)
alerts = _load_alerts(con)
g = build_graph()

report = structural_report(g)
detections = _run_changepoint_backtest(con, incidents, alerts)
candidates = _run_propagation_mining(con, incidents)
validation = validate_against_dependency_graph(candidates, g, max_hops=2, enrichment_threshold=1.5)
observation_end = incidents["started_at"].max() if len(incidents) else pd.Timestamp.now()
curves = reliability_curves_all_services(incidents, observation_end)

with st.sidebar:
    st.markdown(theme.eyebrow_html("Structural controls"), unsafe_allow_html=True)
    st.title("Structural Analytics")
    st.caption("Dependency graph is fixed topology (relplatform/generator/graph.py); backtest and mining parameters are set in code (ChangePointConfig, enrichment_scores).")

st.markdown(
    f'''<div class="hero">
      {theme.eyebrow_html("Phase 4 · Structural Analytics")}
      <h1 class="hero-title">How failure moves through the system.</h1>
      <div class="hero-meta">{len(g.nodes())} services · {len(g.edges())} call-graph edges · {len(incidents)} incidents analyzed</div>
    </div>''',
    unsafe_allow_html=True,
)

# ---------------- Blast radius & criticality ----------------
with st.container(border=True):
    theme.panel_header("Dependency Graph", "Blast Radius and Criticality",
                        "Blast radius = services affected if this one fails · criticality = PageRank on the call graph", accent="blue")
    report_df = pd.DataFrame(report)
    st.dataframe(
        report_df[["service", "tier", "blast_radius_count", "criticality_pagerank", "direct_callers"]],
        width="stretch", hide_index=True,
    )
    st.caption("Services affected if this one fails")
    st.bar_chart(report_df.set_index("service")["blast_radius_count"], color="#5EC8F2")

# ---------------- Propagation mining ----------------
with st.container(border=True):
    theme.panel_header("Propagation", "Mined Failure-Propagation Edges",
                        "Incident co-occurrence far more often than chance would predict", accent="violet")
    theme.assumption_note(
        "This generator does not simulate cross-service incident cascades (only ALERTS "
        "propagate across services during a storm; each incident is generated "
        "independently, scoped to one service). So co-occurrence mined below reflects "
        "shared risk factors or chance, not a simulated ground-truth cascade. Precision/"
        "recall compare mined pairs against the REAL dependency graph, which is the "
        "honest thing to check given that limitation -- not proof of causation."
    )
    if len(candidates) == 0:
        st.caption("No service pairs met the minimum co-occurrence bar in the current dataset.")
    else:
        st.dataframe(
            candidates.head(15)[["service_a", "service_b", "observed_count", "expected_count", "enrichment"]]
            .style.format({"expected_count": "{:.2f}", "enrichment": "{:.1f}x"}),
            width="stretch", hide_index=True,
        )
        p = f"{validation['precision']:.0%}" if validation["precision"] is not None else "n/a"
        r = f"{validation['recall']:.0%}" if validation["recall"] is not None else "n/a"
        st.markdown(
            '<div class="stat-grid">'
            + theme.stat_card("Flagged pairs", f"{validation['n_flagged']}", "enrichment >= 1.5x")
            + theme.stat_card("Precision vs. real graph", p, "within 2 hops")
            + theme.stat_card("Recall vs. real graph", r, "of all graph-adjacent pairs")
            + "</div>",
            unsafe_allow_html=True,
        )

# ---------------- Change-point detection backtest ----------------
with st.container(border=True):
    theme.panel_header("Early Warning", "Change-Point Detection Backtest",
                        "CUSUM on incoming alert rate, backtested against known incident start times", accent="amber")
    theme.assumption_note(
        "Runs on the full alert stream, including background noise, not just "
        "incident-attributed alerts -- in production you don't know in advance which "
        "alerts are 'real'. A detection counts as a match if it lands up to 120 minutes "
        "before, or 10 minutes after, a labeled incident's start (relplatform/structural/"
        "changepoint.py: ChangePointConfig). Lead time is measured against the "
        "recorded incident start, not against when a human actually would have noticed."
    )
    if not detections:
        st.caption("No alert/incident data available yet.")
    else:
        det_df = pd.DataFrame([
            {"service": s, "n_incidents": d.n_incidents, "n_detections": d.n_detections,
             "detection_rate": d.detection_rate, "false_positive_rate": d.false_positive_rate,
             "median_lead_minutes": d.median_lead_minutes}
            for s, d in detections.items()
        ])
        st.dataframe(
            det_df.style.format({
                "detection_rate": lambda v: f"{v:.0%}" if pd.notna(v) else "n/a",
                "false_positive_rate": lambda v: f"{v:.0%}" if pd.notna(v) else "n/a",
                "median_lead_minutes": lambda v: f"{v:.0f} min" if pd.notna(v) else "n/a",
            }),
            width="stretch", hide_index=True,
        )
        rates = [d.detection_rate for d in detections.values() if d.detection_rate is not None]
        fprs = [d.false_positive_rate for d in detections.values() if d.false_positive_rate is not None]
        leads = [d.median_lead_minutes for d in detections.values() if d.median_lead_minutes is not None]
        st.markdown(
            '<div class="stat-grid">'
            + theme.stat_card("Fleet detection rate", f"{sum(rates)/len(rates):.0%}" if rates else "n/a", "mean across services")
            + theme.stat_card("Fleet false-positive rate", f"{sum(fprs)/len(fprs):.0%}" if fprs else "n/a")
            + theme.stat_card("Median lead time", f"{sum(leads)/len(leads):.0f} min" if leads else "n/a", "earlier than the recorded start")
            + "</div>",
            unsafe_allow_html=True,
        )

# ---------------- Reliability curves ----------------
with st.container(border=True):
    theme.panel_header("Reliability", "Time-Between-Failures Curves",
                        "Kaplan-Meier survival estimate; the tail gap to the end of the window is censored, not treated as a failure", accent="coral")
    if not curves:
        st.caption("No incident data available yet.")
    else:
        mtbf_df = pd.DataFrame([
            {"service": s, "n_failures_observed": c.n_events, "mtbf_days": (c.mtbf_minutes / 1440) if c.mtbf_minutes else None,
             "median_survival_days": (c.median_survival_minutes / 1440) if c.median_survival_minutes else None}
            for s, c in curves.items()
        ])
        st.dataframe(
            mtbf_df.style.format({"mtbf_days": "{:.1f}", "median_survival_days": lambda v: f"{v:.1f}" if pd.notna(v) else "not reached"}),
            width="stretch", hide_index=True,
        )
        chart_df = _combine_curves_for_chart(curves)
        chart_df.index = chart_df.index / 1440  # minutes -> days for a readable x-axis
        chart_df.index.name = "days since previous failure"
        st.caption("S(t): probability a service goes longer than t days without another incident")
        st.line_chart(chart_df)
