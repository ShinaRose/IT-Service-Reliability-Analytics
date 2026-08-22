"""RAG retrieval eval: precision@3 against known-similar incident pairs.

Ground truth for "similar" comes from the generator's root_cause_category: two
incidents sharing a root cause are treated as a known-similar pair (the same
definition a human building a retrieval eval set for this data would land on, since
category is exactly the thing that determines whether "resembles INC-1042" is useful
advice). Categories with only one member can't have a similar pair and are excluded
from the denominator.

The query for each test incident is built from ITS OWN alert cluster (the same
`retrieve_for_alert_cluster` path a live incident would use), not its postmortem text --
and the index (`build_incident_index`) never embeds root_cause_category. Both matter:
querying with the postmortem text (which mentions the same symptoms the postmortem
consistently uses per category) or indexing on the category label would let cosine
similarity trivially keyword-match the answer rather than measure genuine symptom
similarity between an alert storm and a past write-up.
"""
from __future__ import annotations

import pandas as pd

from relplatform.ai.rag import build_incident_index, retrieve_for_alert_cluster


def precision_at_k(con, incidents: pd.DataFrame, alerts: pd.DataFrame, k: int = 3, sample_n: int | None = 60, seed: int = 5) -> dict:
    index = build_incident_index(con, incidents, alerts)
    cat_counts = incidents["root_cause_category"].value_counts()
    eligible = incidents[incidents["root_cause_category"].isin(cat_counts[cat_counts >= 2].index)]
    queries = eligible.sample(n=min(sample_n or len(eligible), len(eligible)), random_state=seed) if sample_n else eligible

    per_query = []
    for row in queries.itertuples():
        cluster_alerts = alerts[alerts["incident_id"] == row.id]
        if cluster_alerts.empty:
            continue
        results = retrieve_for_alert_cluster(con, index, cluster_alerts, k=k, exclude_incident_id=row.id)
        true_category = row.root_cause_category
        hits = sum(1 for r in results if r["root_cause_category"] == true_category)
        per_query.append({"incident_id": row.id, "true_category": true_category, "hits": hits, "k": len(results),
                           "retrieved": [r["incident_id"] for r in results]})

    precisions = [pq["hits"] / pq["k"] for pq in per_query if pq["k"] > 0]
    return {
        "precision_at_k": float(sum(precisions) / len(precisions)) if precisions else 0.0,
        "k": k,
        "n_queries": len(per_query),
        "per_query": per_query,
    }
