"""Real-world DORA metrics: the same four official metrics the rest of this app reports
on synthetic data, computed instead from a real public GitHub repo or the user's own
uploaded CSVs. See relplatform/external/ for the connector and CSV-mapping logic --
this page renders results and does no DORA computation of its own; every number comes
from relplatform.analytics.dora's own banding functions, the same ones the synthetic
pipeline uses, so a band means the same thing everywhere in this app.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

try:
    for key in ("RELPLATFORM_PROVIDER", "RELPLATFORM_MONTHS", "GEMINI_API_KEY", "RELPLATFORM_GEMINI_MODEL", "GITHUB_TOKEN"):
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass

import pandas as pd

from relplatform.dashboard import theme
from relplatform.dashboard.data import ensure_connection, load_report
from relplatform.external.csv_upload import compute_uploaded_dora, map_deployments, map_incidents
from relplatform.external.github_dora import GitHubRateLimitError, GitHubRepoError, compute_github_dora, parse_repo_input

st.set_page_config(page_title="Real-World DORA", layout="wide", initial_sidebar_state="expanded")
theme.inject()

DORA_SPECS = [
    ("deployment_frequency", "Deployment Frequency", lambda m: f"{m['value_per_day']}/day"),
    ("lead_time_for_changes", "Lead Time for Changes", lambda m: f"{m['median_hours']}h median"),
    ("change_failure_rate", "Change Failure Rate", lambda m: f"{m['value_pct']}%"),
    ("time_to_restore", "Time to Restore", lambda m: f"{m['median_hours']}h median"),
]


def _render_dora_metrics(dora_like: dict, notes: list[str] | None = None) -> None:
    cols = st.columns(4)
    for col, (key, title, fmt) in zip(cols, DORA_SPECS):
        m = dora_like.get(key)
        with col:
            if m is None:
                st.metric(title, "unavailable")
            else:
                st.metric(title, fmt(m), f"{m['trend']['pct_change']}% ({m['trend']['direction']})")
                st.markdown(theme.band_pill(m["band"]), unsafe_allow_html=True)
    if notes:
        with st.expander("Data-quality notes and judgment calls"):
            for n in notes:
                st.caption(f"• {n}")


with st.sidebar:
    st.markdown(theme.eyebrow_html("Data source"), unsafe_allow_html=True)
    st.title("Real-World DORA")
    source = st.radio(
        "Compute DORA metrics from:",
        ["Synthetic (built-in)", "GitHub repository", "Upload your own CSVs"],
        label_visibility="collapsed",
    )
    st.caption("Every panel on this page is labeled with exactly which of these three produced the numbers shown.")

st.markdown(
    f'''<div class="hero">
      {theme.eyebrow_html("Phase 5 · Real Data")}
      <h1 class="hero-title">The same four metrics, on real data.</h1>
      <div class="hero-meta">Deployment frequency · lead time for changes · change failure rate · time to restore</div>
    </div>''',
    unsafe_allow_html=True,
)

# ==================== Synthetic ====================
if source == "Synthetic (built-in)":
    theme.source_badge("synthetic", "this platform's built-in 12-month simulation")
    con = ensure_connection()
    report = load_report(con)
    with st.container(border=True):
        theme.panel_header("DORA", "Four Keys", "From the synthetic dataset the rest of this app is built on", accent="blue")
        _render_dora_metrics(report["dora_metrics"])

# ==================== GitHub ====================
elif source == "GitHub repository":
    with st.container(border=True):
        theme.panel_header("GitHub Connector", "Compute DORA From a Public Repo",
                            "Deployment = a published Release · lead time = release publish time minus earliest commit since the prior release · change failure = a revert commit in that release's window", accent="blue")
        theme.assumption_note(
            "Every mapping below is a real judgment call, not a guess: repos with zero "
            "releases have no deployment-frequency signal (no fallback to counting merge "
            "commits -- that conflates 'merged' with 'deployed'). Time to restore needs an "
            "incident-issue label you supply; there's no universal convention to guess "
            "from. Unauthenticated GitHub calls are capped at 60/hour -- results are "
            "cached for an hour, and calls per repo are bounded regardless of repo size."
        )
        col1, col2 = st.columns([2, 1])
        repo_input = col1.text_input("Public repo", placeholder="owner/repo or https://github.com/owner/repo")
        incident_label = col2.text_input("Incident-issue label (optional)", placeholder="e.g. incident, outage")
        fetch = st.button("Fetch and compute", type="primary")

        if fetch and repo_input.strip():
            token = os.environ.get("GITHUB_TOKEN") or None
            try:
                owner, repo = parse_repo_input(repo_input)
                con = ensure_connection()
                with st.spinner(f"Fetching {owner}/{repo} from the GitHub API..."):
                    result = compute_github_dora(owner, repo, incident_label.strip() or None, con=con, token=token)
                st.session_state["github_dora_result"] = result
            except GitHubRateLimitError as e:
                st.error(str(e))
                st.session_state.pop("github_dora_result", None)
            except GitHubRepoError as e:
                st.error(str(e))
                st.session_state.pop("github_dora_result", None)

        result = st.session_state.get("github_dora_result")
        if result is not None:
            theme.source_badge("github", f"{result.owner}/{result.repo} · default branch {result.default_branch} · fetched {result.fetched_at}")
            dora_like = {
                "deployment_frequency": result.deployment_frequency,
                "lead_time_for_changes": result.lead_time_for_changes,
                "change_failure_rate": result.change_failure_rate,
                "time_to_restore": result.time_to_restore,
            }
            _render_dora_metrics(dora_like, result.notes)
        else:
            theme.source_badge("github", "no repo fetched yet")
            st.caption("Enter a public owner/repo and click Fetch and compute. Try e.g. 'facebook/react' or 'pallets/flask'.")

# ==================== CSV upload ====================
else:
    with st.container(border=True):
        theme.panel_header("CSV Upload", "Compute DORA From Your Own Data",
                            "Two files: one row per deployment, one row per incident -- map your own column names below", accent="violet")
        theme.assumption_note(
            "Change failure rate reuses this platform's own time-proximity heuristic "
            "(a deploy is charged with an incident if one starts within 4 hours on the "
            "same service, matched to the nearest prior deploy) -- the same judgment call "
            "the synthetic pipeline documents, applied here to your data. If you don't map "
            "a service column, every row is treated as one shared service."
        )

        st.markdown("**Deployments**")
        dep_file = st.file_uploader("deployments.csv -- needs at least a deployment timestamp", type="csv", key="dep_csv")
        dep_mapped = None
        if dep_file is not None:
            raw_dep = pd.read_csv(dep_file)
            cols = list(raw_dep.columns)
            c1, c2, c3, c4 = st.columns(4)
            deployed_at_col = c1.selectbox("deployed_at", cols, key="dep_deployed_at")
            service_col = c2.selectbox("service (optional)", ["(none)"] + cols, key="dep_service")
            lead_time_col = c3.selectbox("lead_time_hours (optional)", ["(none)"] + cols, key="dep_lead_hours")
            commit_at_col = c4.selectbox("commit_at (optional, alt.)", ["(none)"] + cols, key="dep_commit_at")
            dep_mapped = map_deployments(
                raw_dep, deployed_at_col,
                service_col=None if service_col == "(none)" else service_col,
                lead_time_hours_col=None if lead_time_col == "(none)" else lead_time_col,
                commit_at_col=None if commit_at_col == "(none)" else commit_at_col,
            )

        st.markdown("**Incidents**")
        inc_file = st.file_uploader("incidents.csv -- needs a start and resolution timestamp", type="csv", key="inc_csv")
        inc_mapped = None
        if inc_file is not None:
            raw_inc = pd.read_csv(inc_file)
            cols = list(raw_inc.columns)
            c1, c2, c3 = st.columns(3)
            started_at_col = c1.selectbox("started_at", cols, key="inc_started_at")
            resolved_at_col = c2.selectbox("resolved_at", cols, key="inc_resolved_at")
            service_col2 = c3.selectbox("service (optional)", ["(none)"] + cols, key="inc_service")
            inc_mapped = map_incidents(
                raw_inc, started_at_col, resolved_at_col,
                service_col=None if service_col2 == "(none)" else service_col2,
            )

        if dep_file is None and inc_file is None:
            theme.source_badge("uploaded", "no files uploaded yet")
            st.caption("Upload at least one CSV (deployments and/or incidents) to compute DORA metrics from your own data.")
        else:
            label = f"{dep_file.name if dep_file else 'no deployments file'} + {inc_file.name if inc_file else 'no incidents file'}"
            theme.source_badge("uploaded", label)
            dep_mapped = dep_mapped or map_deployments(pd.DataFrame({"deployed_at": pd.Series(dtype="object")}), "deployed_at")
            inc_mapped = inc_mapped or map_incidents(
                pd.DataFrame({"started_at": pd.Series(dtype="object"), "resolved_at": pd.Series(dtype="object")}),
                "started_at", "resolved_at",
            )
            result = compute_uploaded_dora(dep_mapped, inc_mapped)
            dora_like = {
                "deployment_frequency": result.deployment_frequency,
                "lead_time_for_changes": result.lead_time_for_changes,
                "change_failure_rate": result.change_failure_rate,
                "time_to_restore": result.time_to_restore,
            }
            notes = [f"{e.field}: {e.message}" for e in result.errors]
            st.caption(f"{result.n_deployments} deployment row(s), {result.n_incidents} incident row(s) used after validation.")
            _render_dora_metrics(dora_like, notes)
