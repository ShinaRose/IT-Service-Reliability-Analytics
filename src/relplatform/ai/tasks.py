"""AI batch tasks: alert-cluster narrative summarization and postmortem root-cause
categorization. Both are structured_output() calls validated against a JSON schema and
cached by content hash -- the LLM never sees or computes a numeric metric, only text.
"""
from __future__ import annotations

import pandas as pd

from relplatform.ai.cache import cached_structured_call
from relplatform.ai.schemas import NARRATIVE_SCHEMA, ROOT_CAUSE_SCHEMA

NARRATIVE_SYSTEM = (
    "You are an SRE assistant summarizing a storm of correlated monitoring alerts into a "
    "single incident narrative. Be concise and factual; do not invent services or numbers "
    "that are not present in the alert list."
)

ROOT_CAUSE_SYSTEM = (
    "You are an SRE assistant categorizing an incident postmortem. Choose exactly one root "
    "cause category from the allowed enum, list the contributing factors mentioned in the "
    "text, and state the single preventive action recommended."
)


def summarize_alert_cluster(con, provider, cluster_alerts: pd.DataFrame, cluster_id: str) -> tuple[dict, "CachedCallStats"]:
    ordered = cluster_alerts.sort_values("fired_at")
    lines = [f"{i+1}. [{r.fired_at}] service={r.service} severity={r.severity}: {r.message}" for i, r in enumerate(ordered.itertuples())]
    prompt = (
        f"Alert cluster {cluster_id} ({len(ordered)} alerts across {ordered['service'].nunique()} services):\n"
        + "\n".join(lines[:60])  # cap prompt size; storms can be 200 alerts
        + ("\n... (truncated)" if len(lines) > 60 else "")
    )
    return cached_structured_call(con, provider, task="narrative", prompt=prompt, schema=NARRATIVE_SCHEMA, system=NARRATIVE_SYSTEM)


def categorize_postmortem(con, provider, postmortem_text: str, incident_id: str) -> tuple[dict, "CachedCallStats"]:
    prompt = f"Postmortem for {incident_id}:\n\n{postmortem_text}"
    return cached_structured_call(con, provider, task="root_cause", prompt=prompt, schema=ROOT_CAUSE_SCHEMA, system=ROOT_CAUSE_SYSTEM)
