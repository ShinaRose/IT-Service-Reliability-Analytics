from relplatform.reporting.pdf_summary import build_exec_summary_pdf


def _report(risk_scores=None, breaches=None) -> dict:
    return {
        "dora_metrics": {
            "deployment_frequency": {"value_per_day": 10.7, "band": "elite"},
            "lead_time_for_changes": {"median_hours": 1.6, "band": "high"},
            "change_failure_rate": {"value_pct": 9.0, "band": "elite"},
            "time_to_restore": {"median_hours": 0.6, "band": "elite"},
        },
        "risk_scores": risk_scores if risk_scores is not None else [
            {"service": "payments-service", "risk_score": 73.7, "incidents_per_month": 12.0, "mttr_p90_minutes": 90.0},
            {"service": "search-service", "risk_score": 20.1, "incidents_per_month": 2.0, "mttr_p90_minutes": 15.0},
        ],
        "noise_reduction": {"noise_reduction_rate": 0.864, "n_alerts": 44717, "n_distinct_clusters": 6093},
        "capacity_forecasts": breaches if breaches is not None else [
            {"service": "payments-service", "status": "breach_projected"},
            {"service": "search-service", "status": "stable_or_declining"},
        ],
    }


def test_build_exec_summary_pdf_is_a_valid_pdf():
    pdf_bytes = build_exec_summary_pdf(_report(), seed=42)
    assert pdf_bytes[:5] == b"%PDF-"
    assert len(pdf_bytes) > 500


def test_build_exec_summary_pdf_with_ai_narrative():
    pdf_bytes = build_exec_summary_pdf(_report(), seed=42, ai_narrative="Reliability improved this month across the fleet.")
    assert pdf_bytes[:5] == b"%PDF-"


def test_build_exec_summary_pdf_truncates_long_narrative():
    long_text = "x" * 5000
    pdf_bytes = build_exec_summary_pdf(_report(), seed=42, ai_narrative=long_text)
    assert pdf_bytes[:5] == b"%PDF-"


def test_build_exec_summary_pdf_no_risk_scores_or_breaches():
    pdf_bytes = build_exec_summary_pdf(_report(risk_scores=[], breaches=[]), seed=42)
    assert pdf_bytes[:5] == b"%PDF-"
