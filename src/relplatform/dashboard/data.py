"""Shared DuckDB connection + bootstrap + report loading, used by every dashboard page
(Home and pages/). Extracted from the original single-page app.py so a new page doesn't
duplicate the CacheReplayClosureError-avoidance logic (see _bootstrap's docstring).
Every page gets a correctly cache-safe, race-safe connection just by calling
ensure_connection().
"""
from __future__ import annotations

import streamlit as st

from relplatform.bootstrap import ensure_ready
from relplatform.db import get_connection
from relplatform.pipeline import compute_full_report, load_latest_report, persist_report


@st.cache_resource
def _connection():
    return get_connection()


@st.cache_resource
def _bootstrap(_con) -> bool:
    # cache_resource, not a plain call: this must run exactly once per process, not once
    # per script rerun (and, with multiple pages, not once per page either: cache_resource
    # is keyed by function identity, so every page calling this same function shares one
    # result). ensure_ready() is idempotent on its own (a fast no-op once data exists), but
    # idempotent isn't the same as race-free. A page reload during a slow cold start would
    # otherwise trigger a second, overlapping bootstrap before the first has committed any
    # data. Wrapping it in cache_resource makes concurrent reruns share the one in-flight
    # (or completed) call instead of racing.
    #
    # No st.* calls inside this function's body: cache_resource replays a cached function's
    # UI calls on every cache hit, and Streamlit can't safely replay one invoked through a
    # closure passed in as an argument (like `progress=lambda msg: st.toast(msg)` would be).
    # That raises CacheReplayClosureError, which only ever surfaced on a genuinely cold
    # deploy (Streamlit Cloud), never locally.
    ensure_ready(_con, progress=print)
    return True


def ensure_connection():
    """Call this first, on every page. Cheap after the first call in the process (cached)."""
    con = _connection()
    with st.spinner("First run: generating data and computing the report (a few minutes)..."):
        _bootstrap(con)
    return con


def load_report(con, force: bool = False) -> dict:
    if force:
        report = compute_full_report(con)
        persist_report(con, report)
        return report
    report = load_latest_report(con)
    if report is None:
        report = compute_full_report(con)
        persist_report(con, report)
    return report
