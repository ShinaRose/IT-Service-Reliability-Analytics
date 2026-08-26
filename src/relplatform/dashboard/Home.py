"""Streamlit exec dashboard. Home page. Run with: streamlit run src/relplatform/dashboard/Home.py

Named Home.py, not app.py: Streamlit's multipage sidebar labels the entry script by its
filename, and "app.py" showed up in the nav as the unhelpful literal word "app". This is
the only reason for the name; nothing else about it is special.

On a fresh environment (e.g. a Streamlit Community Cloud deploy, which never receives
the git-ignored 145MB reliability.duckdb) this bootstraps itself: generates data,
clusters alerts, and computes the report on first load. Locally, if you've already run
the scripts/ pipeline, that work is reused as-is.

This is a Streamlit multipage app: this file is "Home", and pages/ holds one script per
extension phase (SLOs, financial, on-call, ...). Streamlit only executes the page
currently being viewed, which is what gives lazy-per-tab loading on a memory-constrained
deploy. Each page calls relplatform.dashboard.data.ensure_connection() independently
rather than this file eagerly loading everything every other page might need.

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
# exists at all, which is the normal case for local development. Every page repeats this
# block. Redundant if Home ran first in this process, but necessary if a user deep-links
# straight to another page on a fresh session (Streamlit then runs that page's script
# first, not this one).
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
from relplatform.config import RANDOM_SEED
from relplatform.dashboard import theme
from relplatform.dashboard.data import ensure_connection, load_report
from relplatform.reporting.pdf_summary import build_exec_summary_pdf

st.set_page_config(page_title="Reliability Analytics", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
theme.inject()

band_pill = theme.band_pill
eyebrow_html = theme.eyebrow_html
panel_header = theme.panel_header
stat_card = theme.stat_card


@st.cache_data
def _load_tables(_con):
    # incidents: only ever used on this page for its distinct `service` values, so this
    # narrows out postmortem_text (a full generated paragraph per row) rather than
    # pulling the whole table for nothing. alerts is NOT loaded here at all: it's the
    # largest table on the whole platform (tens of thousands of rows including message
    # text) and this page only touches it inside the "Recompute clustering" branch,
    # which most visits never trigger -- see _load_alerts() below, called lazily only
    # when that button is actually clicked.
    return {
        "incidents": _con.execute("SELECT id, service FROM incidents").df(),
        "resource_metrics": _con.execute("SELECT * FROM resource_metrics").df(),
        "deploy_risk_scores": _con.execute("SELECT * FROM deploy_risk_scores").df(),
    }


@st.cache_data
def _load_alerts(_con):
    return _con.execute("SELECT id, service, message, fired_at, incident_id FROM alerts").df()


@st.cache_data
def _build_exec_summary_pdf_cached(_report: dict, seed: int, period_label: str, computed_at: str, ai_narrative: str | None) -> bytes:
    # `computed_at`/`ai_narrative` are the PDF's only real inputs (report content is
    # keyed by when it was computed, not re-hashed via `_report` -- the leading
    # underscore tells st.cache_data to skip hashing that argument). Without this, every
    # slider drag on this page (weights, capacity threshold, flagging percentile) reruns
    # the whole script and rebuilds a PDF nobody asked to re-download yet.
    return build_exec_summary_pdf(_report, seed=seed, period_label=period_label, ai_narrative=ai_narrative)


con = ensure_connection()
report = load_report(con, force=False)
tables = _load_tables(con)
all_services = sorted(tables["incidents"]["service"].unique())

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.markdown(eyebrow_html("Live controls"), unsafe_allow_html=True)
    st.title("Reliability Analytics")
    st.caption(f"Report computed {report['computed_at'][:19]} UTC")
    st.caption(f"Synthetic data generated with seed={RANDOM_SEED} (set RELPLATFORM_SEED to change it, then regenerate).")

    if st.button("Recompute full report", width="stretch"):
        report = load_report(con, force=True)
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown('<div class="control-label">Services</div>', unsafe_allow_html=True)
    selected_services = st.multiselect("Services", all_services, default=all_services, label_visibility="collapsed")

    st.divider()
    st.markdown('<div class="control-label">Risk score weights</div>', unsafe_allow_html=True)
    st.caption("Instant reweighting, no recompute. Re-blends the three normalized signals already on disk.")
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
    st.caption("Synthetic data. See the Real-World DORA page to compute these same four metrics from a public GitHub repo or your own uploaded data.")

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
            clustered = cluster_alerts(con, _load_alerts(con), cfg)
            new_noise = noise_reduction_rate(clustered)
            new_eval = evaluate_against_ground_truth(clustered)
        st.success("Recomputed with your parameters below (not persisted; use 'Recompute full report' to make new clustering permanent).")
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
        theme.bar_chart(risk_df_view.set_index("service")["risk_score"], color="#F3B94D")

# ---------------- MTTR ----------------
with st.container(border=True):
    panel_header("Recovery", "MTTR Distribution", "Log-normal / Weibull fit per service, reported as percentiles, not the mean", accent="violet")
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
        # Cutoff is deliberately computed across the whole fleet (deploy_scores), not just
        # the filtered `scoped` set. "Top 10% riskiest" should mean fleet-wide, so that
        # narrowing the Services filter shows how many of *that* service's deploys clear a
        # fixed bar, not a moving one that always flags ~10% of whatever's currently shown.
        # The caption used to read "top X%... N of M in the selected services" without
        # saying the bar itself was fleet-wide, so a filtered view could show e.g. 8 of 12
        # flagged and look broken next to "top 10%". Now it says so explicitly.
        cutoff = deploy_scores["risk_probability"].quantile(flag_pctile)
        flagged = scoped[scoped["risk_probability"] >= cutoff].sort_values("risk_probability", ascending=False)
        st.caption(f"Flagging deploys at or above the fleet-wide top {100 * (1 - flag_pctile):.0f}% risk bar "
                   f"(probability ≥ {cutoff:.3f}). {len(flagged)} of {len(scoped)} deploys in the selected "
                   f"services clear that bar.")
        st.dataframe(flagged.head(25)[["id", "service", "deployed_at", "risk_probability"]],
                     width="stretch", hide_index=True)
    else:
        st.info("No persisted deploy risk scores for the selected services yet. Click 'Recompute full report'.")

# ---------------- Exec summary ----------------
with st.container(border=True):
    panel_header("AI layer", "Monthly Exec Summary", "Generated, not computed. Every number in it has to trace back to the report")
    if not provider_ready:
        st.warning("AI provider unavailable. Set RELPLATFORM_PROVIDER (mock/ollama/gemini) and, for gemini, GEMINI_API_KEY.")
    else:
        if st.button("Generate exec summary"):
            try:
                with st.spinner("Generating..."):
                    # risk_df_view (filtered by the Services multiselect), not risk_df.
                    # Every other live panel on this page respects the filter, and the
                    # summary silently ranking services the user excluded from view would
                    # look inconsistent with everything else on screen.
                    text, context, stats, unsupported_numbers = generate_exec_summary(
                        con, provider, dora, nr, risk_df_view, report["capacity_forecasts"], "current period",
                    )
                if unsupported_numbers:
                    st.warning(f"Could not fully verify this summary against the report after retrying. "
                               f"It may state numbers not present in the input: {unsupported_numbers}")
                st.session_state["exec_summary_text"] = text
                st.caption(f"tokens_in={stats.tokens_in} tokens_out={stats.tokens_out} cache_hit={stats.hit}")
            except Exception as e:
                st.error(f"Generation failed: {e}. If using Ollama, it isn't reachable from this "
                         f"environment (Ollama only runs locally). Switch RELPLATFORM_PROVIDER to "
                         f"'gemini' or 'mock'.")

        if st.session_state.get("exec_summary_text"):
            st.markdown(st.session_state["exec_summary_text"])

    st.divider()
    st.caption("One-page PDF: DORA metrics, top risk services, noise reduction, and capacity outlook, "
               "plus the AI narrative above if you've generated one.")
    pdf_bytes = _build_exec_summary_pdf_cached(
        report, RANDOM_SEED, "current period", report["computed_at"],
        st.session_state.get("exec_summary_text"),
    )
    st.download_button(
        "Download exec summary (PDF)", data=pdf_bytes, file_name="reliability_exec_summary.pdf",
        mime="application/pdf",
    )
