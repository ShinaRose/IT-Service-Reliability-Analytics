"""Streamlit exec dashboard. Run with: streamlit run src/relplatform/dashboard/app.py

On a fresh environment (e.g. a Streamlit Community Cloud deploy, which never receives
the git-ignored 145MB reliability.duckdb) this bootstraps itself: generates data,
clusters alerts, and computes the report on first load. Locally, if you've already run
the scripts/ pipeline, that work is reused as-is.

Interactive controls live in the sidebar. Risk-score weights, the capacity breach
threshold, and the change-failure flagging percentile recompute instantly (cheap
pandas/scipy operations over already-persisted numbers); clustering parameters recompute
behind an explicit button since a re-cluster costs a few seconds, not milliseconds.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

# Streamlit secrets -> env vars, BEFORE importing relplatform.config (which reads env
# vars at import time). Wrapped in try/except: st.secrets raises if no secrets.toml
# exists at all, which is the normal case for local development.
try:
    for key in ("RELPLATFORM_PROVIDER", "RELPLATFORM_MONTHS", "GEMINI_API_KEY", "RELPLATFORM_GEMINI_MODEL"):
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass

import pandas as pd

from relplatform.ai.exec_summary import generate_exec_summary
from relplatform.ai.provider import get_provider
from relplatform.analytics.capacity import forecast_all_services
from relplatform.analytics.clustering import ClusteringConfig, cluster_alerts, evaluate_against_ground_truth, noise_reduction_rate
from relplatform.bootstrap import ensure_ready
from relplatform.db import get_connection
from relplatform.pipeline import compute_full_report, load_latest_report, persist_report

st.set_page_config(page_title="Reliability Analytics", layout="wide", initial_sidebar_state="expanded")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --rp-bg: #0A0E16;
  --rp-surface: #121926;
  --rp-surface-2: #171F31;
  --rp-border: #232C40;
  --rp-accent: #3FD9C7;
  --rp-accent-dim: #1F5F58;
  --rp-ok: #4ADE94;    --rp-ok-soft: #10261B;
  --rp-warn: #F3B94D;  --rp-warn-soft: #2B2210;
  --rp-bad: #F1706B;   --rp-bad-soft: #2C1414;
  --rp-text: #EDF1F5;
  --rp-text-dim: #9AA7B8;
  --rp-text-faint: #5C6880;

  /* Per-panel signature hues -- each major section gets its own color so the page reads
     as distinct zones at a glance, not one accent repeated seven times. Not arbitrary:
     blue for the clustering/data-processing panel, amber for priority (reusing the same
     amber as the "medium" DORA band -- it already means "pay attention" on this page),
     violet for recovery, coral for a forward-looking forecast, rose for the risk/failure
     domain (reusing the "bad" semantic color, since that IS this panel's subject). DORA
     and the AI exec summary bookend the page in the primary teal on purpose. */
  --rp-blue: #5EC8F2;   --rp-blue-soft: #0E2530;
  --rp-violet: #B18CF5; --rp-violet-soft: #211A33;
  --rp-coral: #FF9166;  --rp-coral-soft: #2C1B10;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif; letter-spacing: -0.01em; }

/* Page background: a faint accent glow in the corner, not a loud gradient hero */
[data-testid="stAppViewContainer"], .stApp {
  background:
    radial-gradient(1100px 560px at 12% -8%, rgba(63,217,199,0.055), transparent 60%),
    var(--rp-bg) !important;
}
[data-testid="stHeader"] { background: transparent !important; }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--rp-bg); }
::-webkit-scrollbar-thumb { background: var(--rp-border); border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: var(--rp-accent-dim); }

/* ---------------- Eyebrow label (reused everywhere: hero, panels, sidebar) ---------------- */
.eyebrow {
  font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; letter-spacing: 0.09em;
  text-transform: uppercase; color: var(--rp-accent); display: flex; align-items: center; gap: 7px;
  margin-bottom: 8px;
}
.eyebrow .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--rp-accent); flex-shrink: 0; }

/* ---------------- Hero ---------------- */
.hero { padding: 4px 0 30px; }
.hero-title {
  font-family: 'IBM Plex Sans', sans-serif; font-weight: 700; font-size: clamp(26px, 3.2vw, 38px);
  letter-spacing: -0.02em; line-height: 1.18; color: var(--rp-text); margin: 0 0 12px; max-width: 24ch;
}
.hero-meta { font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; color: var(--rp-text-faint); }

/* ---------------- Headline stat grid ---------------- */
.stat-grid {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px;
  background: var(--rp-border); border: 1px solid var(--rp-border); border-radius: 14px;
  overflow: hidden; margin-bottom: 30px;
}
.stat-card { background: var(--rp-surface); padding: 20px 22px; transition: background 0.15s ease; }
.stat-card:hover { background: var(--rp-surface-2); }
.stat-value {
  font-family: 'IBM Plex Mono', monospace; font-size: 25px; font-weight: 600;
  color: var(--rp-accent); letter-spacing: -0.01em; line-height: 1.2;
}
.stat-label { font-size: 12.5px; color: var(--rp-text-faint); margin-top: 6px; }
.stat-sub { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--rp-text-dim); margin-top: 4px; }

/* ---------------- Panels (st.container(border=True)) ----------------
   Streamlit's own bordered-container testid isn't stable across versions (this one
   doesn't emit stVerticalBlockBorderWrapper at all). Anchored instead to our own
   .panel-head marker, which panel_header() renders as the first element inside every
   real panel: `stLayoutWrapper > stVerticalBlock:has(.panel-head)` matches exactly the
   6-7 panels that call panel_header() and nothing else (column cells and the page's
   outermost block share the same testid pair but never contain that marker). */
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(.panel-head) {
  background: var(--rp-surface) !important;
  border: 1px solid var(--rp-border) !important;
  border-radius: 14px !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.25), 0 16px 32px -22px rgba(0,0,0,0.6);
  padding: 20px 22px !important;
  position: relative;
}
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(.panel-head)::before {
  content: ""; position: absolute; top: -1px; left: 18px; right: 18px; height: 2px;
  background: linear-gradient(90deg, var(--rp-accent), transparent 85%);
  border-radius: 2px; opacity: 0.65; pointer-events: none;
}
div[data-testid="stLayoutWrapper"]:has(.panel-head) { margin-bottom: 20px; }

/* Per-panel accent overrides -- same :has() anchoring, scoped by an accent-* class on
   the panel's own .panel-head so each panel's top bar + eyebrow pick up its signature
   hue instead of the default teal. */
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(.panel-head.accent-blue)::before   { background: linear-gradient(90deg, var(--rp-blue), transparent 85%); }
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(.panel-head.accent-amber)::before  { background: linear-gradient(90deg, var(--rp-warn), transparent 85%); }
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(.panel-head.accent-violet)::before { background: linear-gradient(90deg, var(--rp-violet), transparent 85%); }
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(.panel-head.accent-coral)::before  { background: linear-gradient(90deg, var(--rp-coral), transparent 85%); }
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(.panel-head.accent-rose)::before   { background: linear-gradient(90deg, var(--rp-bad), transparent 85%); }

.panel-head.accent-blue .eyebrow   { color: var(--rp-blue); }   .panel-head.accent-blue .eyebrow .dot   { background: var(--rp-blue); }
.panel-head.accent-amber .eyebrow  { color: var(--rp-warn); }   .panel-head.accent-amber .eyebrow .dot  { background: var(--rp-warn); }
.panel-head.accent-violet .eyebrow { color: var(--rp-violet); } .panel-head.accent-violet .eyebrow .dot { background: var(--rp-violet); }
.panel-head.accent-coral .eyebrow  { color: var(--rp-coral); }  .panel-head.accent-coral .eyebrow .dot  { background: var(--rp-coral); }
.panel-head.accent-rose .eyebrow   { color: var(--rp-bad); }    .panel-head.accent-rose .eyebrow .dot   { background: var(--rp-bad); }

.panel-head { display: flex; align-items: flex-end; justify-content: space-between; gap: 14px; margin-bottom: 14px; flex-wrap: wrap; }
.panel-title { font-family: 'IBM Plex Sans', sans-serif; font-weight: 600; font-size: 17px; letter-spacing: -0.01em; color: var(--rp-text); margin: 0; }
.panel-note { font-size: 12.5px; color: var(--rp-text-faint); text-align: right; max-width: 42ch; }

/* ---------------- Metrics ---------------- */
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 1.5rem; color: var(--rp-text); }
[data-testid="stMetricLabel"] { color: var(--rp-text-faint) !important; font-size: 12.5px; }
[data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 12.5px; }

/* ---------------- Band pills ---------------- */
.band-pill {
  display: inline-flex; align-items: center; gap: 5px;
  font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 600;
  letter-spacing: 0.03em; text-transform: uppercase;
  padding: 3px 10px; border-radius: 999px; margin-top: 4px;
}
.band-elite  { background: var(--rp-ok-soft);   color: var(--rp-ok); }
.band-high   { background: rgba(63,217,199,0.12); color: var(--rp-accent); }
.band-medium { background: var(--rp-warn-soft); color: var(--rp-warn); }
.band-low    { background: var(--rp-bad-soft);  color: var(--rp-bad); }

/* ---------------- Sidebar ---------------- */
section[data-testid="stSidebar"] { background: var(--rp-surface); border-right: 1px solid var(--rp-border); }
section[data-testid="stSidebar"] h1 { font-size: 19px !important; margin-top: 0; }
section[data-testid="stSidebar"] hr { border-color: var(--rp-border); margin: 16px 0; }
.control-label {
  font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; letter-spacing: 0.07em;
  text-transform: uppercase; color: var(--rp-text-dim); margin: 2px 0 0;
}

/* ---------------- Buttons ---------------- */
div[data-testid="stButton"] > button {
  border: 1px solid var(--rp-accent); color: var(--rp-accent); background: transparent;
  border-radius: 8px; font-weight: 500; transition: background-color 0.15s ease, box-shadow 0.15s ease, color 0.15s ease;
}
div[data-testid="stButton"] > button:hover {
  border-color: var(--rp-accent); color: #05110F; background-color: var(--rp-accent);
  box-shadow: 0 6px 20px -6px rgba(63,217,199,0.45);
}
div[data-testid="stButton"] > button:focus:not(:active) { border-color: var(--rp-accent); color: var(--rp-accent); }

/* ---------------- Multiselect chips ---------------- */
[data-testid="stMultiSelectTagsContainer"] span[data-tag] {
  background-color: rgba(63,217,199,0.14) !important; border: 1px solid rgba(63,217,199,0.4) !important;
  border-radius: 6px !important;
}

/* ---------------- Dataframe / expander shells ---------------- */
[data-testid="stDataFrame"], [data-testid="stExpander"] {
  border: 1px solid var(--rp-border) !important; border-radius: 10px !important; overflow: hidden;
}
[data-testid="stExpander"] summary { font-family: 'IBM Plex Sans', sans-serif; }

@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def band_pill(band: str) -> str:
    return f'<span class="band-pill band-{band}">{band}</span>'


def eyebrow_html(text: str) -> str:
    return f'<div class="eyebrow"><span class="dot"></span>{text}</div>'


def panel_header(eyebrow_text: str, title: str, note: str = "", accent: str = "") -> None:
    """accent: '' (teal, the default/brand color) or one of blue/amber/violet/coral/rose --
    see the accent-* CSS rules for what each panel's signature hue is used for."""
    note_html = f'<div class="panel-note">{note}</div>' if note else ""
    accent_cls = f" accent-{accent}" if accent else ""
    st.markdown(
        f'<div class="panel-head{accent_cls}"><div>{eyebrow_html(eyebrow_text)}'
        f'<h3 class="panel-title">{title}</h3></div>{note_html}</div>',
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="stat-sub">{sub}</div>' if sub else ""
    return f'<div class="stat-card"><div class="stat-value">{value}</div><div class="stat-label">{label}</div>{sub_html}</div>'


@st.cache_resource
def _connection():
    return get_connection()


@st.cache_resource
def _bootstrap(_con) -> bool:
    # cache_resource, not a plain call: this must run exactly once per process, not once
    # per script rerun. ensure_ready() is idempotent on its own (a fast no-op once data
    # exists), but idempotent isn't the same as race-free -- a page reload during a slow
    # cold start would otherwise trigger a second, overlapping bootstrap before the first
    # has committed any data. Wrapping it in cache_resource makes concurrent reruns share
    # the one in-flight (or completed) call instead of racing.
    #
    # No st.* calls inside this function's body, unlike the naive version of this fix:
    # cache_resource replays a cached function's UI calls on every cache hit, and Streamlit
    # can't safely replay one invoked through a closure passed in as an argument (like
    # `progress=lambda msg: st.toast(msg)` would be) -- that raises CacheReplayClosureError,
    # which only ever surfaced on a genuinely cold deploy (Streamlit Cloud), never locally.
    ensure_ready(_con, progress=print)
    return True


@st.cache_data
def _load_tables(_con):
    return {
        "incidents": _con.execute("SELECT * FROM incidents").df(),
        "resource_metrics": _con.execute("SELECT * FROM resource_metrics").df(),
        "alerts": _con.execute("SELECT id, service, message, fired_at, incident_id FROM alerts").df(),
        "deploy_risk_scores": _con.execute("SELECT * FROM deploy_risk_scores").df(),
    }


def _load_report(con, force: bool):
    if force:
        report = compute_full_report(con)
        persist_report(con, report)
        return report
    report = load_latest_report(con)
    if report is None:
        report = compute_full_report(con)
        persist_report(con, report)
    return report


con = _connection()
with st.spinner("First run: generating data and computing the report (a few minutes)..."):
    _bootstrap(con)
report = _load_report(con, force=False)
tables = _load_tables(con)
all_services = sorted(tables["incidents"]["service"].unique())

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.markdown(eyebrow_html("Live controls"), unsafe_allow_html=True)
    st.title("Reliability Analytics")
    st.caption(f"Report computed {report['computed_at'][:19]} UTC")

    if st.button("Recompute full report", width="stretch"):
        report = _load_report(con, force=True)
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown('<div class="control-label">Services</div>', unsafe_allow_html=True)
    selected_services = st.multiselect("Services", all_services, default=all_services, label_visibility="collapsed")

    st.divider()
    st.markdown('<div class="control-label">Risk score weights</div>', unsafe_allow_html=True)
    st.caption("Reweight instantly -- no recompute needed, this just re-blends the three normalized signals already on disk.")
    w_freq = st.slider("Incident frequency", 0, 100, 33)
    w_mttr = st.slider("MTTR p90", 0, 100, 33)
    w_cfr = st.slider("Change failure rate", 0, 100, 34)
    w_total = max(1, w_freq + w_mttr + w_cfr)

    st.divider()
    st.markdown('<div class="control-label">Capacity forecast</div>', unsafe_allow_html=True)
    cap_threshold = st.slider("Breach threshold (%)", 50, 99, 90)

    st.divider()
    st.markdown('<div class="control-label">Change-failure flagging</div>', unsafe_allow_html=True)
    flag_pctile = st.slider("High-risk percentile", 50, 99, 90) / 100.0

    st.divider()
    with st.expander("Clustering parameters (costs a few seconds to apply)"):
        window_minutes = st.slider("Rolling window (min)", 5, 60, 20)
        eps = st.slider("DBSCAN eps (cosine distance)", 0.10, 0.60, 0.35, step=0.01)
        min_samples = st.slider("DBSCAN min_samples", 2, 15, 4)
        recluster = st.button("Recompute clustering", width="stretch")

    st.divider()
    try:
        provider = get_provider()
        provider_ready = True
        st.caption(f"AI provider: **{provider.name}** / {provider.model}")
    except Exception as e:
        provider = None
        provider_ready = False
        st.caption(f"AI provider unavailable: {e}")

dora = report["dora_metrics"]
nr = report["noise_reduction"]
ce = report["clustering_ground_truth_eval"]

# ---------------- Live-reweighted risk ranking ----------------
risk_df = pd.DataFrame(report["risk_scores"]).copy()
risk_df["risk_score"] = (
    w_freq / w_total * risk_df["norm_incident_frequency"]
    + w_mttr / w_total * risk_df["norm_mttr_p90"]
    + w_cfr / w_total * risk_df["norm_change_failure_rate"]
) * 100
risk_df = risk_df.sort_values("risk_score", ascending=False)
risk_df_view = risk_df[risk_df["service"].isin(selected_services)] if selected_services else risk_df.iloc[0:0]
top_risk = risk_df_view.iloc[0] if len(risk_df_view) else None

# ---------------- Hero ----------------
st.markdown(
    f'''<div class="hero">
      {eyebrow_html("Service Reliability Analytics")}
      <h1 class="hero-title">Where to spend engineering effort, backed by numbers.</h1>
      <div class="hero-meta">{len(all_services)} services · report computed {report['computed_at'][:19]} UTC · live and recomputable</div>
    </div>''',
    unsafe_allow_html=True,
)

# ---------------- Headline strip ----------------
elite_count = sum(1 for m in dora.values() if m["band"] == "elite")
cv_auc = report["change_failure_model"]["metrics"].get("cv_roc_auc_mean", float("nan"))
st.markdown(
    '<div class="stat-grid">'
    + stat_card("Alert noise reduction", f"{nr['noise_reduction_rate']*100:.1f}%")
    + stat_card(
        "Top risk service",
        top_risk["service"].removesuffix("-service") if top_risk is not None else "n/a",
        f"{top_risk['risk_score']:.1f}/100" if top_risk is not None else "",
    )
    + stat_card("Change-failure model AUC", f"{cv_auc:.3f}")
    + stat_card("DORA metrics at Elite", f"{elite_count} / 4")
    + "</div>",
    unsafe_allow_html=True,
)

# ---------------- DORA ----------------
with st.container(border=True):
    panel_header("Metrics", "DORA Metrics", "Official four-keys definitions · Elite/High/Medium/Low bands from Accelerate")
    cols = st.columns(4)
    labels = {
        "deployment_frequency": ("Deployment Frequency", lambda m: f"{m['value_per_day']}/day"),
        "lead_time_for_changes": ("Lead Time for Changes", lambda m: f"{m['median_hours']}h median"),
        "change_failure_rate": ("Change Failure Rate", lambda m: f"{m['value_pct']}%"),
        "time_to_restore": ("Time to Restore", lambda m: f"{m['median_hours']}h median"),
    }
    for col, (key, (title, fmt)) in zip(cols, labels.items()):
        m = dora[key]
        with col:
            st.metric(title, fmt(m), f"{m['trend']['pct_change']}% ({m['trend']['direction']})")
            st.markdown(band_pill(m["band"]), unsafe_allow_html=True)

# ---------------- Alert deduplication ----------------
with st.container(border=True):
    panel_header("Clustering", "Alert Deduplication", "Rolling time window + service-dependency blocking, then DBSCAN on message embeddings", accent="blue")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw alerts", f"{nr['n_alerts']:,}")
    c2.metric("Distinct clusters", f"{nr['n_distinct_clusters']:,}")
    c3.metric("Adjusted Rand Index", f"{ce['adjusted_rand_index']:.3f}")
    c4.metric("Purity", f"{ce['purity']:.3f}")

    if recluster:
        with st.spinner(f"Re-clustering: window={window_minutes}min, eps={eps}, min_samples={min_samples}..."):
            cfg = ClusteringConfig(window_minutes=float(window_minutes), dbscan_eps=float(eps), dbscan_min_samples=int(min_samples))
            clustered = cluster_alerts(con, tables["alerts"], cfg)
            new_noise = noise_reduction_rate(clustered)
            new_eval = evaluate_against_ground_truth(clustered)
        st.success("Recomputed with your parameters below (not persisted -- use 'Recompute full report' to make new clustering permanent).")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Raw alerts", f"{new_noise['n_alerts']:,}")
        d2.metric("Distinct clusters", f"{new_noise['n_distinct_clusters']:,}",
                   delta=f"{new_noise['n_distinct_clusters'] - nr['n_distinct_clusters']:+,}", delta_color="inverse")
        d3.metric("Adjusted Rand Index", f"{new_eval['adjusted_rand_index']:.3f}",
                   delta=f"{new_eval['adjusted_rand_index'] - ce['adjusted_rand_index']:+.3f}")
        d4.metric("Purity", f"{new_eval['purity']:.3f}", delta=f"{new_eval['purity'] - ce['purity']:+.3f}")

# ---------------- Risk ranking ----------------
with st.container(border=True):
    panel_header("Priority", "Service Risk Ranking",
                 f"Weights: frequency {w_freq/w_total:.0%} · MTTR p90 {w_mttr/w_total:.0%} · change failure {w_cfr/w_total:.0%}",
                 accent="amber")
    st.dataframe(
        risk_df_view[["service", "risk_score", "incidents_per_month", "mttr_p90_minutes", "change_failure_rate"]]
        .style.format({"risk_score": "{:.1f}", "incidents_per_month": "{:.2f}", "mttr_p90_minutes": "{:.1f}",
                        "change_failure_rate": "{:.1%}"}),
        width="stretch", hide_index=True,
    )
    if len(risk_df_view):
        st.bar_chart(risk_df_view.set_index("service")["risk_score"], color="#F3B94D")

# ---------------- MTTR ----------------
with st.container(border=True):
    panel_header("Recovery", "MTTR Distribution", "Log-normal / Weibull fit per service, reported as percentiles -- not the mean", accent="violet")
    mttr_rows = []
    for svc, fit in report["mttr_fits"].items():
        if svc not in selected_services:
            continue
        pct = fit.get("fitted_percentiles_minutes") or fit.get("empirical_percentiles", {})
        mttr_rows.append({"service": svc, **{k: pct.get(k) for k in ["p50", "p75", "p90", "p95", "p99"]}})
    st.dataframe(pd.DataFrame(mttr_rows), width="stretch", hide_index=True)

# ---------------- Capacity forecast ----------------
with st.container(border=True):
    panel_header("Forecast", "Capacity Forecast",
                 f"Threshold {cap_threshold}% · gated on p&lt;0.05 and r²≥0.10 (a positive slope alone isn't a trend)",
                 accent="coral")
    live_capacity = forecast_all_services(tables["resource_metrics"], threshold=float(cap_threshold))
    cap_df = pd.DataFrame(live_capacity)
    cap_df = cap_df[cap_df["service"].isin(selected_services)] if selected_services else cap_df.iloc[0:0]
    st.dataframe(cap_df, width="stretch", hide_index=True)

# ---------------- Change failure model ----------------
with st.container(border=True):
    panel_header("Pre-merge risk", "Change-Failure Model", "Logistic regression on deploy features, flagging risky changes before merge", accent="rose")
    cf = report["change_failure_model"]
    st.write(f"5-fold CV ROC AUC: **{cf['metrics'].get('cv_roc_auc_mean', float('nan')):.3f}** "
             f"(± {cf['metrics'].get('cv_roc_auc_std', 0):.3f})")
    coef_df = pd.DataFrame(list(cf["coefficients"].items()), columns=["feature", "coefficient"]).head(8)
    st.dataframe(coef_df, width="stretch", hide_index=True)

    deploy_scores = tables["deploy_risk_scores"].copy()
    scoped = deploy_scores[deploy_scores["service"].isin(selected_services)] if selected_services else deploy_scores.iloc[0:0]
    if len(scoped):
        cutoff = deploy_scores["risk_probability"].quantile(flag_pctile)
        flagged = scoped[scoped["risk_probability"] >= cutoff].sort_values("risk_probability", ascending=False)
        st.caption(f"Flagging the top {100 * (1 - flag_pctile):.0f}% by predicted risk "
                   f"(probability ≥ {cutoff:.3f}) -- {len(flagged)} of {len(scoped)} deploys in the selected services")
        st.dataframe(flagged.head(25)[["id", "service", "deployed_at", "risk_probability"]],
                     width="stretch", hide_index=True)
    else:
        st.info("No persisted deploy risk scores for the selected services yet -- click 'Recompute full report'.")

# ---------------- Exec summary ----------------
with st.container(border=True):
    panel_header("AI layer", "Monthly Exec Summary", "Generated, not computed -- every number in it has to trace back to the report")
    if not provider_ready:
        st.warning("AI provider unavailable. Set RELPLATFORM_PROVIDER (mock/ollama/gemini) and, for gemini, GEMINI_API_KEY.")
    else:
        if st.button("Generate exec summary"):
            try:
                with st.spinner("Generating..."):
                    text, context, stats = generate_exec_summary(
                        con, provider, dora, nr, risk_df, report["capacity_forecasts"], "current period",
                    )
                st.markdown(text)
                st.caption(f"tokens_in={stats.tokens_in} tokens_out={stats.tokens_out} cache_hit={stats.hit}")
            except Exception as e:
                st.error(f"Generation failed: {e}. If using Ollama, it isn't reachable from this "
                         f"environment (Ollama only runs locally) -- switch RELPLATFORM_PROVIDER to "
                         f"'gemini' or 'mock'.")
