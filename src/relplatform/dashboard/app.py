"""Streamlit exec dashboard. Run with: streamlit run src/relplatform/dashboard/app.py
Reads the precomputed report from DuckDB (run scripts/run_pipeline.py first)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from relplatform.ai.exec_summary import generate_exec_summary
from relplatform.ai.provider import get_provider
from relplatform.db import get_connection
from relplatform.pipeline import compute_full_report, load_latest_report, persist_report

st.set_page_config(page_title="Reliability Analytics", layout="wide")


@st.cache_resource
def _connection():
    return get_connection()


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

st.title("Service Reliability Analytics")
if st.button("Recompute report"):
    report = _load_report(con, force=True)
    st.success(f"Recomputed at {report['computed_at']}")
else:
    report = _load_report(con, force=False)

st.caption(f"Report computed at {report['computed_at']}")

# ---- DORA ----
st.header("DORA Metrics")
dora = report["dora_metrics"]
cols = st.columns(4)
band_color = {"elite": "🟢", "high": "🟡", "medium": "🟠", "low": "🔴"}
labels = {
    "deployment_frequency": ("Deployment Frequency", lambda m: f"{m['value_per_day']}/day"),
    "lead_time_for_changes": ("Lead Time for Changes", lambda m: f"{m['median_hours']}h median"),
    "change_failure_rate": ("Change Failure Rate", lambda m: f"{m['value_pct']}%"),
    "time_to_restore": ("Time to Restore", lambda m: f"{m['median_hours']}h median"),
}
for col, (key, (title, fmt)) in zip(cols, labels.items()):
    m = dora[key]
    col.metric(title, fmt(m), f"{m['trend']['pct_change']}% ({m['trend']['direction']})")
    col.caption(f"{band_color.get(m['band'],'')} {m['band'].upper()} band")

# ---- Noise reduction / clustering ----
st.header("Alert Noise Reduction")
nr = report["noise_reduction"]
ce = report["clustering_ground_truth_eval"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Noise reduction rate", f"{nr['noise_reduction_rate']*100:.1f}%")
c2.metric("Raw alerts -> clusters", f"{nr['n_alerts']:,} -> {nr['n_distinct_clusters']:,}")
c3.metric("Adjusted Rand Index", f"{ce['adjusted_rand_index']:.3f}")
c4.metric("Purity", f"{ce['purity']:.3f}")

# ---- Risk ranking ----
st.header("Service Risk Ranking (where to spend engineering effort)")
risk_df = pd.DataFrame(report["risk_scores"])
st.dataframe(
    risk_df[["service", "risk_score", "incidents_per_month", "mttr_p90_minutes", "change_failure_rate"]]
    .style.format({"risk_score": "{:.1f}", "incidents_per_month": "{:.2f}", "mttr_p90_minutes": "{:.1f}",
                    "change_failure_rate": "{:.1%}"}),
    use_container_width=True,
)
st.bar_chart(risk_df.set_index("service")["risk_score"])

# ---- MTTR ----
st.header("MTTR Distribution (percentiles, per service)")
mttr_rows = []
for svc, fit in report["mttr_fits"].items():
    pct = fit.get("fitted_percentiles_minutes") or fit.get("empirical_percentiles", {})
    mttr_rows.append({"service": svc, **{k: pct.get(k) for k in ["p50", "p75", "p90", "p95", "p99"]}})
st.dataframe(pd.DataFrame(mttr_rows), use_container_width=True)

# ---- Capacity forecast ----
st.header("Capacity Forecast")
cap_rows = report["capacity_forecasts"]
st.dataframe(pd.DataFrame(cap_rows), use_container_width=True)

# ---- Change failure model ----
st.header("Change Failure Model")
cf = report["change_failure_model"]
st.write(f"5-fold CV ROC AUC: **{cf['metrics'].get('cv_roc_auc_mean', float('nan')):.3f}** "
         f"(± {cf['metrics'].get('cv_roc_auc_std', 0):.3f})")
st.write("Top coefficients (why a deploy is flagged risky):")
coef_df = pd.DataFrame(list(cf["coefficients"].items()), columns=["feature", "coefficient"]).head(8)
st.dataframe(coef_df, use_container_width=True)
st.write("Recently flagged high-risk deploys:")
st.dataframe(pd.DataFrame(cf["top_flagged_deploys"]), use_container_width=True)

# ---- Exec summary ----
st.header("Monthly Exec Summary (AI-generated, numbers-checked)")
provider = get_provider()
st.caption(f"Provider: {provider.name} / {provider.model}")
if st.button("Generate exec summary"):
    text, context, stats = generate_exec_summary(
        con, provider, dora, nr, risk_df, cap_rows, "current period",
    )
    st.markdown(text)
    st.caption(f"tokens_in={stats.tokens_in} tokens_out={stats.tokens_out} cache_hit={stats.hit}")
