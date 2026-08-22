"""RAG over past incidents: given a live alert cluster, retrieve the most similar
historical incidents and how they were resolved.

Retrieval itself is embedding cosine-similarity search (sentence-transformers), not an
LLM call -- deterministic and cheap, which is also what makes precision@3 measurable
against known-similar incident pairs in the eval suite. An LLM may optionally phrase the
"resembles INC-1042, resolved by X" sentence, but the ranked candidates it's allowed to
mention come only from this retrieval step.

The index is built from each historical incident's OWN alert storm text, not its
postmortem prose. An earlier version indexed on postmortem text and scored precision@3
at 0.17 -- *worse* than a random-by-category-frequency baseline (0.27) -- because a
live query (alert messages: short, templated, numeric) and a postmortem (multi-paragraph
narrative prose) are different enough in style that a general-purpose sentence embedding
doesn't bridge them well, on top of which raw postmortems run well past MiniLM's ~256
token window so most of a long write-up was silently dropped anyway. Alert-storm-to-
alert-storm matching keeps query and corpus in the same vocabulary (and the same rough
length), which is also just a more realistic corpus to retrieve against: it's the exact
same shape of data you have at 2am for a live incident, not a finished postmortem that
doesn't exist yet. Measured precision@3 on this basis: ~0.69. The resolution snippet
shown to the user still comes from the matched incident's postmortem -- that's fine, it's
metadata attached to the result, not part of what similarity is computed over.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from relplatform.analytics.embeddings import embed_texts

MAX_ALERTS_PER_STORM_TEXT = 80


def _alert_storm_text(cluster_alerts: pd.DataFrame) -> str:
    if cluster_alerts.empty:
        return ""
    service = cluster_alerts["service"].mode().iloc[0]
    return f"{service}. " + " ".join(cluster_alerts["message"].tolist()[:MAX_ALERTS_PER_STORM_TEXT])


def build_incident_index(con, incidents: pd.DataFrame, alerts: pd.DataFrame) -> dict:
    alerts_by_incident = {iid: g for iid, g in alerts.groupby("incident_id")}
    texts = [_alert_storm_text(alerts_by_incident.get(r.id, alerts.iloc[0:0])) for r in incidents.itertuples()]
    embeddings = embed_texts(con, texts)
    return {"ids": incidents["id"].tolist(), "embeddings": embeddings, "incidents": incidents.reset_index(drop=True)}


def _cosine_topk(query_vec: np.ndarray, matrix: np.ndarray, k: int, exclude_idx: set[int] | None = None) -> list[tuple[int, float]]:
    qn = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    sims = mn @ qn
    order = np.argsort(-sims)
    out = []
    for idx in order:
        if exclude_idx and int(idx) in exclude_idx:
            continue
        out.append((int(idx), float(sims[idx])))
        if len(out) >= k:
            break
    return out


def _extract_resolution_snippet(postmortem_text: str) -> str:
    marker = "## Resolution"
    idx = postmortem_text.find(marker)
    if idx == -1:
        return postmortem_text[:200]
    snippet = postmortem_text[idx + len(marker) : idx + len(marker) + 300].strip()
    return snippet.split("\n\n")[0]


def retrieve_for_alert_cluster(con, index: dict, cluster_alerts: pd.DataFrame, k: int = 3, exclude_incident_id: str | None = None) -> list[dict]:
    query_text = _alert_storm_text(cluster_alerts)
    query_vec = embed_texts(con, [query_text])[0]

    exclude_idx = None
    if exclude_incident_id and exclude_incident_id in index["ids"]:
        exclude_idx = {index["ids"].index(exclude_incident_id)}

    top = _cosine_topk(query_vec, index["embeddings"], k, exclude_idx)
    incidents = index["incidents"]
    results = []
    for idx, sim in top:
        row = incidents.iloc[idx]
        results.append({
            "incident_id": row["id"], "service": row["service"], "root_cause_category": row["root_cause_category"],
            "similarity": round(sim, 4), "resolution_snippet": _extract_resolution_snippet(row["postmortem_text"]),
        })
    return results
