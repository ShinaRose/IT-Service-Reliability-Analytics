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

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 1.6rem; color: #3FD9C7; }
[data-testid="stMetricLabel"] { color: #A9B4C2; }
[data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace !important; }

h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif; letter-spacing: -0.01em; }

.band-pill {
  display: inline-flex; align-items: center; gap: 5px;
  font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 600;
  letter-spacing: 0.03em; text-transform: uppercase;
  padding: 3px 10px; border-radius: 999px; margin-top: 2px;
}
.band-elite { background: #10261B; color: #4ADE94; }
.band-high { background: #142A27; color: #3FD9C7; }
.band-medium { background: #2B2210; color: #F3B94D; }
.band-low { background: #2C1414; color: #F1706B; }

.sidebar-eyebrow {
  font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.08em;
  text-transform: uppercase; color: #3FD9C7; margin-bottom: 4px;
}

div[data-testid="stButton"] > button { border-color: #3FD9C7; color: #3FD9C7; }
div[data-testid="stButton"] > button:hover { border-color: #3FD9C7; color: #0B0F16; background-color: #3FD9C7; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def band_pill(band: str) -> str:
    return f'<span class="band-pill band-{band}">{band}</span>'


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
    st.markdown('<div class="sidebar-eyebrow">● Live controls</div>', unsafe_allow_html=True)
    st.title("Reliability Analytics")
    st.caption(f"Report computed {report['computed_at'][:19]} UTC")

    if st.button("Recompute full report", width="stretch"):
        report = _load_report(con, force=True)
        st.cache_data.clear()
        st.rerun()

    st.divider()
    selected_services = st.multiselect("Services", all_services, default=all_services)

    st.divider()
    st.subheader("Risk score weights")
    st.caption("Reweight instantly -- no recompute needed, this just re-blends the three normalized signals already on disk.")
    w_freq = st.slider("Incident frequency", 0, 100, 33)
    w_mttr = st.slider("MTTR p90", 0, 100, 33)
    w_cfr = st.slider("Change failure rate", 0, 100, 34)
    w_total = max(1, w_freq + w_mttr + w_cfr)

    st.divider()
    st.subheader("Capacity forecast")
    cap_threshold = st.slider("Breach threshold (%)", 50, 99, 90)

    st.divider()
    st.subheader("Change-failure flagging")
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

st.title("Service Reliability Analytics")

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

# ---------------- Headline strip ----------------
hcols = st.columns([1, 1.3, 1, 1])
hcols[0].metric("Alert noise reduction", f"{nr['noise_reduction_rate']*100:.1f}%")
hcols[1].metric("Top risk service", top_risk["service"].removesuffix("-service") if top_risk is not None else "n/a",
                 f"{top_risk['risk_score']:.1f}/100" if top_risk is not None else None)
hcols[2].metric("Change-failure model AUC", f"{report['change_failure_model']['metrics'].get('cv_roc_auc_mean', float('nan')):.3f}")
elite_count = sum(1 for m in dora.values() if m["band"] == "elite")
hcols[3].metric("DORA metrics at Elite", f"{elite_count} / 4")

st.divider()

# ---------------- DORA ----------------
with st.container(border=True):
    st.subheader("DORA Metrics")
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
    st.subheader("Alert Deduplication")
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
    st.subheader("Service Risk Ranking (where to spend engineering effort)")
    st.caption(f"Weights: incident frequency {w_freq/w_total:.0%} · MTTR p90 {w_mttr/w_total:.0%} · "
               f"change failure rate {w_cfr/w_total:.0%}")
    st.dataframe(
        risk_df_view[["service", "risk_score", "incidents_per_month", "mttr_p90_minutes", "change_failure_rate"]]
        .style.format({"risk_score": "{:.1f}", "incidents_per_month": "{:.2f}", "mttr_p90_minutes": "{:.1f}",
                        "change_failure_rate": "{:.1%}"}),
        width="stretch", hide_index=True,
    )
    if len(risk_df_view):
        st.bar_chart(risk_df_view.set_index("service")["risk_score"], color="#3FD9C7")

# ---------------- MTTR ----------------
with st.container(border=True):
    st.subheader("MTTR Distribution (percentiles, per service)")
    mttr_rows = []
    for svc, fit in report["mttr_fits"].items():
        if svc not in selected_services:
            continue
        pct = fit.get("fitted_percentiles_minutes") or fit.get("empirical_percentiles", {})
        mttr_rows.append({"service": svc, **{k: pct.get(k) for k in ["p50", "p75", "p90", "p95", "p99"]}})
    st.dataframe(pd.DataFrame(mttr_rows), width="stretch", hide_index=True)

# ---------------- Capacity forecast ----------------
with st.container(border=True):
    st.subheader("Capacity Forecast")
    st.caption(f"Threshold: {cap_threshold}% · gated on p<0.05 and r²≥0.10 (a positive slope alone isn't a trend)")
    live_capacity = forecast_all_services(tables["resource_metrics"], threshold=float(cap_threshold))
    cap_df = pd.DataFrame(live_capacity)
    cap_df = cap_df[cap_df["service"].isin(selected_services)] if selected_services else cap_df.iloc[0:0]
    st.dataframe(cap_df, width="stretch", hide_index=True)

# ---------------- Change failure model ----------------
with st.container(border=True):
    st.subheader("Change-Failure Model")
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
    st.subheader("Monthly Exec Summary (AI-generated, numbers-checked)")
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
