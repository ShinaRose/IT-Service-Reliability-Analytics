"""On-call health: pages per shift, out-of-hours/sleep-hours load, interrupt
concentration, and alert fatigue by service. See relplatform/oncall/ for the underlying
math. This page renders it and does no computation of its own beyond formatting.
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
from relplatform.oncall.config import load_oncall_config
from relplatform.oncall.fatigue import alert_fatigue_by_service
from relplatform.oncall.pages import (
    assign_pages_to_shifts,
    interrupt_concentration,
    out_of_hours_rate,
    pages_per_shift,
    pages_per_shift_percentiles,
    sleep_hours_interruptions,
)

st.set_page_config(page_title="On-Call Health", layout="wide", initial_sidebar_state="expanded")
theme.inject()


@st.cache_data
def _load_incidents(_con):
    df = _con.execute("SELECT id, service, started_at FROM incidents").df()
    df["started_at"] = pd.to_datetime(df["started_at"])
    return df


@st.cache_data
def _load_shifts(_con):
    df = _con.execute("SELECT id, engineer, shift_start, shift_end, is_holiday, swapped FROM on_call_shifts").df()
    df["shift_start"] = pd.to_datetime(df["shift_start"])
    df["shift_end"] = pd.to_datetime(df["shift_end"])
    return df


@st.cache_data
def _load_clustered_alerts(_con):
    row = _con.execute("SELECT count(*) FROM alert_clusters").fetchone()
    if not row or row[0] == 0:
        return pd.DataFrame(columns=["service", "cluster_id"])
    return _con.execute(
        "SELECT a.service, c.cluster_id FROM alerts a JOIN alert_clusters c ON a.id = c.alert_id"
    ).df()


con = ensure_connection()
oncall_cfg = load_oncall_config()
incidents = _load_incidents(con)
shifts = _load_shifts(con)
clustered_alerts = _load_clustered_alerts(con)

paged = assign_pages_to_shifts(incidents, shifts)
n_holiday_shifts = int(shifts["is_holiday"].sum()) if len(shifts) else 0
n_swapped = int(shifts["swapped"].sum()) if len(shifts) else 0

with st.sidebar:
    st.markdown(theme.eyebrow_html("On-call controls"), unsafe_allow_html=True)
    st.title("On-Call Health")
    st.caption("Windows come from config/oncall.yaml. Edit that file to change them.")
    st.markdown('<div class="control-label">Business hours</div>', unsafe_allow_html=True)
    st.caption(f"{oncall_cfg.business_start_hour:02d}:00–{oncall_cfg.business_end_hour:02d}:00, weekdays {oncall_cfg.business_weekdays}")
    st.markdown('<div class="control-label">Sleep hours</div>', unsafe_allow_html=True)
    st.caption(f"{oncall_cfg.sleep_start_hour:02d}:00–{oncall_cfg.sleep_end_hour:02d}:00, every day")

st.markdown(
    f'''<div class="hero">
      {theme.eyebrow_html("Phase 3 · On-Call Health")}
      <h1 class="hero-title">Who's carrying the pager, and how hard.</h1>
      <div class="hero-meta">{len(shifts)} shifts · {len(incidents)} pages · {n_holiday_shifts} holiday shifts ({n_swapped} swapped)</div>
    </div>''',
    unsafe_allow_html=True,
)

theme.assumption_note(
    "Modeled as a single org-wide primary rotation covering all 8 services, not a "
    "per-service rotation, the common setup for a team this size. Every incident is "
    "one page to whoever's shift covers its start time; the dozens of alerts in that "
    "incident's storm are not separately paged (real paging tools coalesce a storm into "
    "one notification, same as this platform's alert-dedup clustering recovers "
    "computationally). See src/relplatform/generator/roster.py."
)

# ---------------- Pages per shift ----------------
with st.container(border=True):
    theme.panel_header("Pages", "Pages per Shift", "Distribution across all shifts, including quiet ones. Percentiles, not a mean", accent="blue")
    pct = pages_per_shift_percentiles(paged, shifts)

    if pct["n_shifts"] == 0:
        st.caption("No shifts in the current dataset.")
    else:
        st.markdown(
            '<div class="stat-grid">'
            + theme.stat_card("Median (p50)", f"{pct['percentiles']['p50']:.1f}", "pages/shift")
            + theme.stat_card("p90", f"{pct['percentiles']['p90']:.1f}", "pages/shift")
            + theme.stat_card("p99", f"{pct['percentiles']['p99']:.1f}", "pages/shift")
            + theme.stat_card("Worst shift", f"{pct['max']}", "pages, single shift")
            + "</div>",
            unsafe_allow_html=True,
        )
        by_shift = pages_per_shift(paged)
        by_engineer = by_shift.groupby("engineer")["n_pages"].sum().sort_values(ascending=False)
        st.caption("Total pages by engineer, across all their shifts")
        theme.bar_chart(by_engineer, color="#5EC8F2")

# ---------------- Out-of-hours / sleep-hours ----------------
with st.container(border=True):
    theme.panel_header("Interruptions", "Out-of-Hours and Sleep-Hours Load",
                        "Two different questions: inconvenient vs. actually woke someone up", accent="amber")
    theme.assumption_note(
        "Out-of-hours = outside the configured business-hours window (any weekday hour "
        "outside it, or any weekend hour). Sleep-hours is a separate, narrower window "
        "checked independently, not derived as out-of-hours' complement. 19:00 on a "
        "weekday is inconvenient but doesn't wake anyone up."
    )
    ooh = out_of_hours_rate(paged, oncall_cfg)
    sleep = sleep_hours_interruptions(paged, oncall_cfg)

    col1, col2 = st.columns(2)
    with col1:
        if ooh["out_of_hours_rate"] is None:
            st.caption("No paged incidents.")
        else:
            st.metric("Out-of-hours page rate", f"{ooh['out_of_hours_rate']:.1%}", f"{ooh['n_out_of_hours']} of {ooh['n_pages']} pages")
    with col2:
        if sleep["sleep_hours_rate"] is None:
            st.caption("No paged incidents.")
        else:
            st.metric("Sleep-hours interruptions", f"{sleep['sleep_hours_rate']:.1%}", f"{sleep['n_sleep_hours']} of {sleep['n_pages']} pages")

# ---------------- Interrupt concentration ----------------
with st.container(border=True):
    theme.panel_header("Concentration", "Interrupt Load Concentration",
                        "Gini coefficient of pages-per-engineer: 0 = perfectly even, closer to 1 = dumped on one or two people", accent="violet")
    conc = interrupt_concentration(paged)
    if conc["n_engineers"] == 0:
        st.caption("No paged incidents.")
    else:
        st.markdown(
            '<div class="stat-grid">'
            + theme.stat_card("Gini coefficient", f"{conc['gini']:.2f}")
            + theme.stat_card("Top-1 engineer's share", f"{conc['top1_share']:.1%}", "of all pages")
            + theme.stat_card("Engineers paged", f"{conc['n_engineers']}")
            + "</div>",
            unsafe_allow_html=True,
        )
        conc_df = pd.DataFrame(conc["by_engineer"]).set_index("engineer")
        theme.bar_chart(conc_df["n_pages"], color="#B18CF5")

# ---------------- Alert fatigue by service ----------------
with st.container(border=True):
    theme.panel_header("Fatigue", "Alert Fatigue Score, by Service",
                        "50% alert noise ratio + 50% paging load, each min-max normalized across services", accent="coral")
    theme.assumption_note(
        "The 50/50 weighting between noise ratio (raw alerts per distinct clustered "
        "incident) and paging load (pages/month) is a stated choice, not a derived "
        "constant. Edit relplatform/oncall/fatigue.py's NOISE_WEIGHT/PAGE_LOAD_WEIGHT "
        "to change the balance. Scores are relative to this fleet's own 8 services "
        "(min-max normalized), not an absolute scale."
    )
    if len(shifts):
        months = max(1.0, (shifts["shift_end"].max() - shifts["shift_start"].min()).days / 30.0)
    else:
        months = 1.0
    fatigue = alert_fatigue_by_service(clustered_alerts, paged, months=months)
    if len(fatigue) == 0:
        st.caption("No clustered alert data available yet.")
    else:
        st.dataframe(
            fatigue[["service", "raw_alerts", "distinct_clusters", "noise_ratio", "pages_per_month", "fatigue_score"]]
            .style.format({"noise_ratio": "{:.1f}", "pages_per_month": "{:.1f}", "fatigue_score": "{:.0f}"}),
            width="stretch", hide_index=True,
        )
        theme.bar_chart(fatigue.set_index("service")["fatigue_score"], color="#FF9166")
