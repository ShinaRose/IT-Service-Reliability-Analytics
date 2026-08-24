"""Service dependency graph as a first-class analytics object: blast radius (who is
affected if a service goes down) and PageRank-style criticality (how structurally
central a service is). Built on the same networkx graph the generator uses to spread
alert storms (relplatform.generator.graph.build_graph) -- one source of truth for the
topology, not a second copy that could silently drift from what the simulator actually
does. relplatform.analytics.clustering already imports this same graph for the same
reason.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from relplatform.generator.graph import build_graph


@dataclass
class BlastRadius:
    service: str
    direct_callers: list[str]     # services one hop away that call this one directly
    affected_services: list[str]  # full transitive closure: everyone who breaks if this fails
    blast_radius_count: int


def blast_radius(g: nx.DiGraph | None = None) -> dict[str, BlastRadius]:
    """For each service, the set of OTHER services that transitively depend on it (its
    callers, and their callers, ...) -- i.e. who is affected if this service goes down.
    Edges point caller -> callee, so this is nx.ancestors(g, service): every node with a
    directed path TO `service`."""
    g = g if g is not None else build_graph()
    results = {}
    for service in g.nodes():
        ancestors = nx.ancestors(g, service)
        results[service] = BlastRadius(
            service=service,
            direct_callers=sorted(g.predecessors(service)),
            affected_services=sorted(ancestors),
            blast_radius_count=len(ancestors),
        )
    return results


def criticality_scores(g: nx.DiGraph | None = None) -> dict[str, float]:
    """PageRank over the call graph as-is (edges caller -> callee): a service accumulates
    rank from every service that calls it, so a widely-depended-upon backend service
    (many callers, and/or callers that are themselves important) scores highest -- the
    same "important nodes get pointed to by other important nodes" intuition PageRank
    uses for web pages, applied to service call graphs."""
    g = g if g is not None else build_graph()
    return nx.pagerank(g)


def structural_report(g: nx.DiGraph | None = None) -> list[dict]:
    """One row per service combining blast radius and criticality, sorted by
    blast_radius_count descending -- the dashboard's primary table."""
    g = g if g is not None else build_graph()
    br = blast_radius(g)
    crit = criticality_scores(g)
    rows = [
        {
            "service": service,
            "tier": g.nodes[service].get("tier"),
            "blast_radius_count": br[service].blast_radius_count,
            "affected_services": br[service].affected_services,
            "direct_callers": br[service].direct_callers,
            "criticality_pagerank": round(crit[service], 4),
        }
        for service in g.nodes()
    ]
    rows.sort(key=lambda r: (r["blast_radius_count"], r["criticality_pagerank"]), reverse=True)
    return rows
