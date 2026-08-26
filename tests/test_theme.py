import networkx as nx

from relplatform.dashboard.theme import render_dependency_graph_svg
from relplatform.structural.graph import structural_report


def _chain_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(["A", "B", "C"], tier="core")
    g.add_edges_from([("A", "B"), ("B", "C")])
    return g


def test_render_dependency_graph_svg_is_valid_svg():
    g = _chain_graph()
    report = structural_report(g)
    svg = render_dependency_graph_svg(g, report)
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")


def test_render_dependency_graph_svg_node_count_matches_graph():
    g = _chain_graph()
    report = structural_report(g)
    svg = render_dependency_graph_svg(g, report)
    assert svg.count("<circle") == g.number_of_nodes() == 3


def test_render_dependency_graph_svg_edge_count_matches_graph():
    g = _chain_graph()
    report = structural_report(g)
    svg = render_dependency_graph_svg(g, report)
    assert svg.count("<line") == g.number_of_edges() == 2


def test_render_dependency_graph_svg_handles_isolated_node_no_edges():
    g = nx.DiGraph()
    g.add_node("solo", tier="core")
    report = structural_report(g)
    svg = render_dependency_graph_svg(g, report)
    assert svg.count("<circle") == 1
    assert svg.count("<line") == 0


def test_render_dependency_graph_svg_real_service_graph():
    from relplatform.generator.graph import build_graph

    g = build_graph()
    report = structural_report(g)
    svg = render_dependency_graph_svg(g, report)
    assert svg.count("<circle") == 8
    assert svg.count("<line") == len(g.edges())
    # every service label (short form) should appear in the SVG text
    for service in g.nodes():
        assert service.removesuffix("-service") in svg
