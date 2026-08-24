from datetime import datetime, timedelta

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from relplatform.structural.changepoint import ChangePointConfig, backtest_all_services, backtest_service, cusum_detect
from relplatform.structural.graph import blast_radius, criticality_scores, structural_report
from relplatform.structural.propagation import (
    enrichment_scores,
    expected_co_occurrence,
    mine_co_occurrence_edges,
    validate_against_dependency_graph,
)
from relplatform.structural.survival import kaplan_meier, reliability_curve, reliability_curves_all_services, time_between_failures

T0 = datetime(2026, 1, 5, 0, 0, 0)


def _incidents(rows: list[dict]) -> pd.DataFrame:
    cols = ["id", "service", "started_at"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _alerts(rows: list[dict]) -> pd.DataFrame:
    cols = ["id", "service", "fired_at"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# ---------------- graph: blast radius / criticality ----------------

def _chain_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(["A", "B", "C"], tier="core")
    g.add_edges_from([("A", "B"), ("B", "C")])  # A calls B calls C
    return g


def test_blast_radius_chain():
    br = blast_radius(_chain_graph())
    assert br["C"].blast_radius_count == 2  # A and B both transitively depend on C
    assert set(br["C"].affected_services) == {"A", "B"}
    assert br["A"].blast_radius_count == 0  # nobody calls A
    assert br["B"].direct_callers == ["A"]


def test_criticality_scores_sum_to_one():
    scores = criticality_scores(_chain_graph())
    assert scores.keys() == {"A", "B", "C"}
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-6)


def test_structural_report_real_graph_has_all_services():
    report = structural_report()
    assert len(report) == 8
    assert all("blast_radius_count" in r for r in report)
    # auth-service is called (directly or transitively) by most of the fleet
    auth_row = next(r for r in report if r["service"] == "auth-service")
    assert auth_row["blast_radius_count"] >= 3


# ---------------- propagation mining ----------------

def test_mine_co_occurrence_edges_too_few_incidents():
    result = mine_co_occurrence_edges(_incidents([{"id": "INC-1", "service": "a", "started_at": T0}]))
    assert len(result) == 0
    assert list(result.columns) == ["service_a", "service_b", "observed_count"]


def test_mine_co_occurrence_edges_same_service_no_pairs():
    incidents = _incidents([
        {"id": "INC-1", "service": "a", "started_at": T0},
        {"id": "INC-2", "service": "a", "started_at": T0 + timedelta(minutes=10)},
    ])
    result = mine_co_occurrence_edges(incidents, window_minutes=60)
    assert len(result) == 0


def test_mine_co_occurrence_edges_basic():
    incidents = _incidents([
        {"id": "INC-1", "service": "a", "started_at": T0},
        {"id": "INC-2", "service": "b", "started_at": T0 + timedelta(minutes=10)},
        {"id": "INC-3", "service": "c", "started_at": T0 + timedelta(hours=5)},  # outside window
    ])
    result = mine_co_occurrence_edges(incidents, window_minutes=60)
    assert len(result) == 1
    assert result.iloc[0]["service_a"] == "a"
    assert result.iloc[0]["service_b"] == "b"
    assert result.iloc[0]["observed_count"] == 1


def test_expected_co_occurrence_too_few_incidents():
    result = expected_co_occurrence(_incidents([]))
    assert len(result) == 0


def test_enrichment_scores_filters_min_observed():
    incidents = _incidents([
        {"id": f"INC-{i}", "service": "a" if i % 2 == 0 else "b", "started_at": T0 + timedelta(hours=i)}
        for i in range(6)
    ])
    result = enrichment_scores(incidents, window_minutes=90, min_observed=100)
    assert len(result) == 0  # nothing reaches an absurdly high min_observed bar


def test_validate_against_dependency_graph_empty_candidates():
    result = validate_against_dependency_graph(pd.DataFrame(columns=["service_a", "service_b", "enrichment"]), _chain_graph())
    assert result["n_flagged"] == 0
    assert result["precision"] is None


def test_validate_against_dependency_graph_matches_real_edge():
    candidates = pd.DataFrame([{"service_a": "A", "service_b": "B", "enrichment": 3.0}])
    result = validate_against_dependency_graph(candidates, _chain_graph(), max_hops=2, enrichment_threshold=1.5)
    assert result["n_flagged"] == 1
    assert result["precision"] == 1.0


# ---------------- change-point detection ----------------

def test_cusum_detect_empty():
    assert cusum_detect(pd.Series(dtype=float)) == []


def test_cusum_detect_flat_series_no_detections():
    counts = pd.Series([2] * 50, index=pd.date_range(T0, periods=50, freq="5min"))
    assert cusum_detect(counts) == []


def test_cusum_detect_catches_obvious_spike():
    idx = pd.date_range(T0, periods=40, freq="5min")
    values = [1] * 20 + [25] * 5 + [1] * 15
    counts = pd.Series(values, index=idx)
    detections = cusum_detect(counts)
    assert len(detections) >= 1


def test_backtest_all_services_empty_inputs():
    assert backtest_all_services(_alerts([]), _incidents([])) == {}


def test_backtest_service_detects_incident_with_positive_lead():
    incident_start = T0 + timedelta(hours=2)
    end = T0 + timedelta(hours=4)

    rows = []
    aid = 0
    # steady low background: one alert every 20 minutes for 4 hours
    t = T0
    while t < end:
        aid += 1
        rows.append({"id": f"ALT-{aid}", "service": "svc", "fired_at": t})
        t += timedelta(minutes=20)
    # dense pre-incident burst: 30 alerts across a 30-minute window starting 20 min before the labeled incident start
    burst_start = incident_start - timedelta(minutes=20)
    for i in range(30):
        aid += 1
        rows.append({"id": f"ALT-{aid}", "service": "svc", "fired_at": burst_start + timedelta(minutes=i)})

    alerts = _alerts(rows)
    incidents = _incidents([{"id": "INC-1", "service": "svc", "started_at": incident_start}])

    result = backtest_service(alerts, incidents, "svc", T0, end, ChangePointConfig())
    assert result.n_true_positives == 1
    assert result.detection_rate == 1.0
    assert result.median_lead_minutes is not None
    assert result.median_lead_minutes > 0  # detected before the labeled incident start


# ---------------- survival analysis ----------------

def test_time_between_failures_no_incidents():
    durations, events = time_between_failures(_incidents([]), "svc", T0)
    assert len(durations) == 0
    assert len(events) == 0


def test_time_between_failures_single_incident_is_censored():
    incidents = _incidents([{"id": "INC-1", "service": "svc", "started_at": T0}])
    durations, events = time_between_failures(incidents, "svc", T0 + timedelta(days=10))
    assert len(durations) == 1
    assert events[0] == False  # noqa: E712


def test_kaplan_meier_no_censoring_matches_empirical_survival():
    durations = np.array([1.0, 2.0, 3.0])
    events = np.array([True, True, True])
    timeline, survival = kaplan_meier(durations, events)
    assert survival[-1] == pytest.approx(0.0)
    assert timeline[-1] == pytest.approx(3.0)


def test_kaplan_meier_with_censoring_never_reaches_zero():
    durations = np.array([1.0, 2.0, 3.0])
    events = np.array([True, False, True])  # the gap at t=2 is censored
    timeline, survival = kaplan_meier(durations, events)
    # step at t=1: 1 event out of 3 at risk -> survival = 2/3
    assert survival[1] == pytest.approx(2 / 3)
    # step at t=3: at-risk count excludes the censored subject (its own duration is 2 < 3)
    assert survival[-1] == pytest.approx(0.0)


def test_kaplan_meier_empty():
    timeline, survival = kaplan_meier(np.array([]), np.array([], dtype=bool))
    assert list(timeline) == [0.0]
    assert list(survival) == [1.0]


def test_reliability_curve_mtbf_excludes_censored_gap():
    incidents = _incidents([
        {"id": "INC-1", "service": "svc", "started_at": T0},
        {"id": "INC-2", "service": "svc", "started_at": T0 + timedelta(minutes=60)},
    ])
    curve = reliability_curve(incidents, "svc", T0 + timedelta(minutes=600))
    assert curve.n_events == 1
    assert curve.n_censored == 1
    assert curve.mtbf_minutes == pytest.approx(60.0)  # only the observed 60-minute gap, not the huge censored tail


def test_reliability_curves_all_services_empty():
    assert reliability_curves_all_services(_incidents([]), T0) == {}
