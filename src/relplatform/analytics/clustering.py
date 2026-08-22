"""Alert deduplication: rolling time-window + service-dependency blocking, then DBSCAN
on message embeddings within each block.

Two stages, both required by the spec:
  1. Coarse blocking -- union-find over alert pairs that are close in time (within a
     rolling `window_minutes`) AND close in the service dependency graph (<= max_hops,
     undirected). This is what "DBSCAN inside a rolling time window, group by service
     dependency" buys you: it keeps the O(n^2) embedding comparison scoped to alerts
     that could plausibly be the same incident, and it lets a single incident's storm
     span more than one window via the union-find transitive closure (a genuine rolling
     window, not a fixed non-overlapping bin).
  2. Fine clustering -- true DBSCAN (cosine distance on MiniLM embeddings) run inside
     each block, so alerts that happen to be time/service-adjacent but describe an
     unrelated problem don't get merged just because they're nearby.

The headline metric is noise reduction: how much the raw alert volume shrinks once
alerts are deduplicated into distinct clusters (each DBSCAN noise point counts as its
own singleton "thing to look at", same as an operator scrolling a dashboard would).
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

from relplatform.analytics.embeddings import embed_texts
from relplatform.generator.graph import build_graph


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _service_hop_lookup(max_hops: int = 2) -> dict[tuple[str, str], int]:
    g = build_graph().to_undirected()
    nodes = list(g.nodes())
    lookup: dict[tuple[str, str], int] = {}
    lengths = dict(nx.all_pairs_shortest_path_length(g, cutoff=max_hops))
    for a in nodes:
        for b, d in lengths.get(a, {}).items():
            lookup[(a, b)] = d
    return lookup


def _service_related(a: str, b: str, lookup: dict[tuple[str, str], int], max_hops: int) -> bool:
    if a == b:
        return True
    return lookup.get((a, b), max_hops + 1) <= max_hops


@dataclass
class ClusteringConfig:
    window_minutes: float = 20.0
    max_hops: int = 2
    dbscan_eps: float = 0.35
    dbscan_min_samples: int = 4


def build_blocks(alerts: pd.DataFrame, cfg: ClusteringConfig) -> np.ndarray:
    """Coarse union-find blocking by rolling time window + service-graph proximity.
    Returns an int array of block ids, same order as `alerts` (already assumed sorted
    by fired_at with a reset integer index)."""
    lookup = _service_hop_lookup(cfg.max_hops)
    n = len(alerts)
    uf = UnionFind(n)
    times = alerts["fired_at"].values.astype("datetime64[s]").astype(np.int64)
    window_s = cfg.window_minutes * 60

    # Service-relatedness only depends on the (small) set of distinct services, so
    # precompute it as a dense boolean matrix once and index into it per-pair with numpy
    # instead of calling _service_related() (dict lookups + branching) O(n * window) times.
    uniq_services = sorted(alerts["service"].unique())
    svc_idx = {s: i for i, s in enumerate(uniq_services)}
    rel = np.array(
        [[_service_related(a, b, lookup, cfg.max_hops) for b in uniq_services] for a in uniq_services]
    )
    service_codes = alerts["service"].map(svc_idx).to_numpy()

    j_start = 0
    for i in range(n):
        while times[i] - times[j_start] > window_s:
            j_start += 1
        if j_start < i:
            js = np.arange(j_start, i)
            related = rel[service_codes[i], service_codes[js]]
            for j in js[related]:
                uf.union(i, int(j))

    roots = np.array([uf.find(i) for i in range(n)])
    _, block_ids = np.unique(roots, return_inverse=True)
    return block_ids


def dbscan_within_blocks(embeddings: np.ndarray, block_ids: np.ndarray, cfg: ClusteringConfig) -> np.ndarray:
    """Runs DBSCAN (cosine distance) inside each coarse block. Returns global cluster
    labels as strings: 'b{block}-c{label}', with each -1 (noise) point given its own
    unique singleton label."""
    labels = np.empty(len(embeddings), dtype=object)
    for block in np.unique(block_ids):
        idx = np.where(block_ids == block)[0]
        if len(idx) == 1:
            labels[idx[0]] = f"b{block}-noise{idx[0]}"
            continue
        sub_emb = embeddings[idx]
        db = DBSCAN(eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples, metric="cosine")
        sub_labels = db.fit_predict(sub_emb)
        for local_i, lbl in zip(idx, sub_labels):
            if lbl == -1:
                labels[local_i] = f"b{block}-noise{local_i}"
            else:
                labels[local_i] = f"b{block}-c{lbl}"
    return labels


def cluster_alerts(con, alerts: pd.DataFrame, cfg: ClusteringConfig | None = None) -> pd.DataFrame:
    """Full pipeline. `alerts` must have columns: id, service, message, fired_at
    (and may include incident_id for later eval -- unused as a clustering feature).
    Returns `alerts` with an added `cluster_id` column."""
    cfg = cfg or ClusteringConfig()
    alerts = alerts.sort_values("fired_at").reset_index(drop=True)
    embeddings = embed_texts(con, alerts["message"].tolist())
    block_ids = build_blocks(alerts, cfg)
    cluster_ids = dbscan_within_blocks(embeddings, block_ids, cfg)
    out = alerts.copy()
    out["cluster_id"] = cluster_ids
    return out


def noise_reduction_rate(clustered: pd.DataFrame) -> dict:
    n_alerts = len(clustered)
    n_clusters = clustered["cluster_id"].nunique()
    n_noise = clustered["cluster_id"].str.contains("-noise").sum()
    n_real_clusters = n_clusters - n_noise + (n_noise > 0) * 0  # noise clusters are singletons, not "grouped"
    return {
        "n_alerts": n_alerts,
        "n_distinct_clusters": n_clusters,     # what an operator has to look at, post-dedup
        "n_noise_singletons": int(n_noise),
        "n_grouped_clusters": int(n_clusters - n_noise),
        "noise_reduction_rate": 1.0 - n_clusters / n_alerts if n_alerts else 0.0,
    }


def evaluate_against_ground_truth(clustered: pd.DataFrame) -> dict:
    """ARI + purity vs. the generator's ground-truth incident_id. Background-noise
    alerts (incident_id is null) are each their own true singleton class, matching how
    a correct clusterer should also leave them as noise/singletons."""
    truth = clustered["incident_id"].copy()
    null_mask = truth.isna()
    truth = truth.astype(object)
    truth[null_mask] = [f"__bg_{i}" for i in clustered.index[null_mask]]

    pred = clustered["cluster_id"]
    ari = adjusted_rand_score(truth.tolist(), pred.tolist())

    # purity = (1/N) * sum over predicted clusters of the size of its majority true class
    # -- `majority_count` below is a count (not a fraction), so it's summed directly and
    # divided by N once, not multiplied by cluster size again.
    df = pd.DataFrame({"truth": truth.values, "pred": pred.values})
    majority_count = df.groupby("pred")["truth"].agg(lambda s: s.value_counts().iloc[0])
    purity = float(majority_count.sum() / len(df))

    return {"adjusted_rand_index": float(ari), "purity": purity}
