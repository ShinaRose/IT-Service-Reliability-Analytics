"""Mines candidate failure-propagation edges from incident co-occurrence: pairs of
services whose incidents happen close together in time far more often than an
independent-Poisson chance model predicts. Cross-validated against the real dependency
graph (generator/graph.py) to check whether what's mined actually recovers known
structure, the same "score it against ground truth" discipline the rest of this
platform applies to clustering (ARI/purity) and the change-failure model (AUC).

Caveat stated up front, not buried: this generator's incidents are NOT simulated as
cross-service cascades -- each incident is generated independently, scoped to a single
`service` (only ALERTS propagate across services during a storm, see
generator/simulate.py). So co-occurrence mined here reflects either genuine shared risk
factors, incidental clustering in time, or chance -- not a simulated ground-truth
incident cascade. The precision/recall below measures agreement with the real dependency
graph, which is the honest thing to check given that limitation, not proof of causation.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

CO_OCCURRENCE_COLUMNS = ["service_a", "service_b", "observed_count"]
EXPECTED_COLUMNS = ["service_a", "service_b", "expected_count"]


def mine_co_occurrence_edges(incidents: pd.DataFrame, window_minutes: float = 60.0) -> pd.DataFrame:
    """Directed edge (service_a -> service_b) counted once per pair of incidents where
    service_a's incident started before service_b's, within `window_minutes` of each
    other. Returns one row per ordered pair with its observed_count."""
    if len(incidents) < 2:
        return pd.DataFrame(columns=CO_OCCURRENCE_COLUMNS)

    df = incidents.sort_values("started_at").reset_index(drop=True)
    times = pd.to_datetime(df["started_at"]).to_numpy(dtype="datetime64[m]").astype(np.int64)
    services = df["service"].to_numpy()
    n = len(df)

    pairs: dict[tuple[str, str], int] = {}
    for i in range(n):
        j = i + 1
        while j < n and (times[j] - times[i]) <= window_minutes:
            if services[j] != services[i]:
                key = (services[i], services[j])
                pairs[key] = pairs.get(key, 0) + 1
            j += 1

    if not pairs:
        return pd.DataFrame(columns=CO_OCCURRENCE_COLUMNS)
    return (
        pd.DataFrame([{"service_a": a, "service_b": b, "observed_count": c} for (a, b), c in pairs.items()])
        .sort_values("observed_count", ascending=False).reset_index(drop=True)
    )


def expected_co_occurrence(incidents: pd.DataFrame, window_minutes: float = 60.0) -> pd.DataFrame:
    """Expected observed_count per ordered service pair under an independent,
    homogeneous-Poisson-process null: for each of service_a's n_a incidents, the chance a
    given OTHER service_b incident lands in the following `window_minutes` is
    (n_b / total_minutes) * window_minutes, so expected(a->b) = n_a * rate_b *
    window_minutes. An approximation (ignores boundary effects at the edges of the
    observation window), fine at this sample size."""
    if len(incidents) < 2:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    total_minutes = (incidents["started_at"].max() - incidents["started_at"].min()).total_seconds() / 60
    total_minutes = max(total_minutes, 1.0)
    counts = incidents.groupby("service").size()
    rates = counts / total_minutes

    rows = [
        {"service_a": a, "service_b": b, "expected_count": float(counts[a] * rates[b] * window_minutes)}
        for a in counts.index for b in counts.index if a != b
    ]
    if not rows:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)
    return pd.DataFrame(rows)


def enrichment_scores(incidents: pd.DataFrame, window_minutes: float = 60.0, min_observed: int = 3) -> pd.DataFrame:
    """observed/expected ratio per pair, filtered to pairs with at least `min_observed`
    co-occurrences (a pair that only ever co-occurred once or twice is too thin a sample
    to call "enriched" regardless of the ratio)."""
    observed = mine_co_occurrence_edges(incidents, window_minutes)
    expected = expected_co_occurrence(incidents, window_minutes)
    if len(expected) == 0:
        return pd.DataFrame(columns=[*EXPECTED_COLUMNS, "observed_count", "enrichment"])

    merged = expected.merge(observed, on=["service_a", "service_b"], how="left")
    merged["observed_count"] = merged["observed_count"].fillna(0).astype(int)
    merged["enrichment"] = merged["observed_count"] / merged["expected_count"].clip(lower=1e-6)
    merged = merged[merged["observed_count"] >= min_observed]
    return merged.sort_values("enrichment", ascending=False).reset_index(drop=True)


def validate_against_dependency_graph(
    candidate_edges: pd.DataFrame, g: nx.DiGraph, max_hops: int = 2, enrichment_threshold: float = 1.5,
) -> dict:
    """precision = fraction of flagged (enrichment >= threshold) pairs that are within
    `max_hops` of each other in the undirected dependency graph. recall = fraction of all
    graph-adjacent (within max_hops) service pairs that got flagged. Both None when
    there's nothing to divide by, not 0 -- an undefined rate is not the same as a bad
    one."""
    und = g.to_undirected()
    services = list(g.nodes())
    lengths = dict(nx.all_pairs_shortest_path_length(und, cutoff=max_hops))

    def adjacent(a: str, b: str) -> bool:
        return b in lengths.get(a, {})

    if "enrichment" not in candidate_edges.columns or len(candidate_edges) == 0:
        flagged = candidate_edges.iloc[0:0]
    else:
        flagged = candidate_edges[candidate_edges["enrichment"] >= enrichment_threshold]

    precision = None
    if len(flagged) > 0:
        matches = flagged.apply(lambda r: adjacent(r["service_a"], r["service_b"]), axis=1)
        precision = float(matches.mean())

    all_adjacent_pairs = {(a, b) for a in services for b in services if a != b and adjacent(a, b)}
    flagged_pairs = set(zip(flagged["service_a"], flagged["service_b"])) if len(flagged) else set()
    recall = (len(flagged_pairs & all_adjacent_pairs) / len(all_adjacent_pairs)) if all_adjacent_pairs else None

    return {
        "n_flagged": int(len(flagged)),
        "n_graph_adjacent_pairs": len(all_adjacent_pairs),
        "precision": precision,
        "recall": recall,
        "note": (
            "Precision/recall against the real dependency graph, not a simulated "
            "propagation ground truth -- this generator doesn't simulate cross-service "
            "incident cascades (see module docstring)."
        ),
    }
