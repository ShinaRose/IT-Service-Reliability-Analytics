"""Mandatory eval test: the exec summary must never state a metric value that isn't
present in the input it was given. See relplatform.ai.numeric_guard for the mechanism
and relplatform.ai.exec_summary for how the input context is built.
"""
from relplatform.ai.exec_summary import build_context, generate_exec_summary
from relplatform.ai.numeric_guard import find_unsupported_numbers
from relplatform.ai.provider import MockProvider


def _fake_metrics():
    dora_metrics = {
        "deployment_frequency": {"value_per_day": 12.3, "value_per_month_median": 320.0, "band": "elite",
                                  "trend": {"direction": "improving", "pct_change": 5.0}},
        "lead_time_for_changes": {"median_hours": 3.4, "band": "high",
                                   "trend": {"direction": "improving", "pct_change": 10.0}},
        "change_failure_rate": {"value_pct": 7.29, "band": "elite",
                                 "trend": {"direction": "worsening", "pct_change": -3.0}},
        "time_to_restore": {"median_hours": 1.8, "band": "high",
                             "trend": {"direction": "flat", "pct_change": 0.5}},
    }
    noise_reduction = {"n_alerts": 39854, "n_distinct_clusters": 5426, "n_grouped_clusters": 924,
                        "n_noise_singletons": 4502, "noise_reduction_rate": 0.8639}
    import pandas as pd

    risk_df = pd.DataFrame([
        {"service": "payments-service", "risk_score": 87.5, "incidents_per_month": 3.1,
         "mttr_p90_minutes": 145.2, "change_failure_rate": 0.11},
        {"service": "checkout-service", "risk_score": 62.0, "incidents_per_month": 2.0,
         "mttr_p90_minutes": 98.4, "change_failure_rate": 0.06},
    ])
    capacity_forecasts = [
        {"service": "payments-service", "metric": "db_connection_pool_utilization_pct", "status": "breach_projected",
         "current_value": 78.4, "threshold": 90.0, "projected_breach_date": "2026-11-02"},
    ]
    return dora_metrics, noise_reduction, risk_df, capacity_forecasts


def test_numeric_guard_passes_when_output_only_cites_input_numbers():
    dora_metrics, noise_reduction, risk_df, capacity_forecasts = _fake_metrics()
    context = build_context(dora_metrics, noise_reduction, risk_df, capacity_forecasts, "August 2026")
    honest_output = (
        "Change failure rate held at 7.29%, in the elite band. Alert noise reduction reached 86.4%, "
        "collapsing 39854 raw alerts into 5426 clusters. payments-service is the top risk area at a "
        "risk score of 87.5 and is projected to breach 90% capacity on 2026-11-02."
    )
    unsupported = find_unsupported_numbers(honest_output, context)
    assert unsupported == []


def test_numeric_guard_catches_fabricated_metric():
    dora_metrics, noise_reduction, risk_df, capacity_forecasts = _fake_metrics()
    context = build_context(dora_metrics, noise_reduction, risk_df, capacity_forecasts, "August 2026")
    hallucinated_output = "Change failure rate improved dramatically to 2.1%, well ahead of target."
    unsupported = find_unsupported_numbers(hallucinated_output, context)
    assert 2.1 in unsupported


def test_exec_summary_generation_cites_only_input_numbers(memdb):
    dora_metrics, noise_reduction, risk_df, capacity_forecasts = _fake_metrics()
    provider = MockProvider()  # schema-free generate() -> falls back to templated text below
    provider.fixtures["Input figures"] = (
        "Reliability held steady this period. Change failure rate was 7.29% (elite band, worsening -3.0%). "
        "Deployment frequency averaged 12.3 per day (elite). Lead time for changes was a median of 3.4 hours (high). "
        "Time to restore was a median of 1.8 hours (high). Alert noise reduction removed 86.4% of raw volume, "
        "collapsing 39854 alerts into 5426 clusters (924 grouped, 4502 singleton). "
        "Top risk: payments-service (risk score 87.5, 3.1 incidents/month, MTTR p90 145.2 min, change failure 11.0%), "
        "projected to breach its 90.0% capacity threshold on 2026-11-02."
    )
    text, context, stats = generate_exec_summary(memdb, provider, dora_metrics, noise_reduction, risk_df, capacity_forecasts, "August 2026")

    unsupported = find_unsupported_numbers(text, context)
    assert unsupported == [], f"exec summary cited numbers not present in its input: {unsupported}"
