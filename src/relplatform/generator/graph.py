"""Service metadata and the call-dependency graph used to propagate alert storms."""
from __future__ import annotations

import networkx as nx

# (service, tier, baseline_risk) -- baseline_risk is a latent 0..1 knob the generator
# uses to make some services more failure-prone than others (feeds the change-failure
# simulation and gives the risk-score model something real to recover).
SERVICE_META = {
    "api-gateway":            {"tier": "edge", "baseline_risk": 0.10, "deploys_per_week": 21},
    "auth-service":            {"tier": "core", "baseline_risk": 0.08, "deploys_per_week": 14},
    "checkout-service":        {"tier": "core", "baseline_risk": 0.18, "deploys_per_week": 5},
    "payments-service":        {"tier": "core", "baseline_risk": 0.22, "deploys_per_week": 1.2},
    "inventory-service":       {"tier": "core", "baseline_risk": 0.12, "deploys_per_week": 3.5},
    "notification-service":    {"tier": "edge", "baseline_risk": 0.06, "deploys_per_week": 28},
    "search-service":          {"tier": "core", "baseline_risk": 0.15, "deploys_per_week": 1.0},
    "recommendation-service":  {"tier": "data", "baseline_risk": 0.20, "deploys_per_week": 0.5},
}

# caller -> callee ("caller depends on callee"). When callee fails, alerts propagate
# upstream to its callers (and their callers), tapering with hop distance.
CALL_EDGES = [
    ("api-gateway", "auth-service"),
    ("api-gateway", "checkout-service"),
    ("api-gateway", "search-service"),
    ("checkout-service", "payments-service"),
    ("checkout-service", "inventory-service"),
    ("checkout-service", "auth-service"),
    ("checkout-service", "notification-service"),
    ("search-service", "recommendation-service"),
    ("payments-service", "auth-service"),
    ("notification-service", "auth-service"),
]


def build_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    for svc, meta in SERVICE_META.items():
        g.add_node(svc, **meta)
    g.add_edges_from(CALL_EDGES)
    return g


def callers_by_hop(g: nx.DiGraph, origin: str, max_hops: int = 2) -> dict[int, list[str]]:
    """Return {hop_distance: [services]} of callers (ancestors) of `origin` up to max_hops,
    i.e. services whose requests transitively depend on `origin` and would see errors
    when `origin` degrades."""
    rev = g.reverse(copy=False)
    hops: dict[int, list[str]] = {}
    frontier = {origin}
    seen = {origin}
    for hop in range(1, max_hops + 1):
        nxt = set()
        for node in frontier:
            nxt |= set(rev.successors(node)) - seen
        if not nxt:
            break
        hops[hop] = sorted(nxt)
        seen |= nxt
        frontier = nxt
    return hops
