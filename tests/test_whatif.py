import pandas as pd
import pytest

from relplatform.finance.whatif import whatif_priced_incidents, whatif_risk_scores


def _priced_incidents() -> pd.DataFrame:
    return pd.DataFrame([
        {"id": "INC-1", "service": "payments-service", "root_cause_category": "deployment_regression",
         "incident_cost_eur": 1000.0, "toil_cost_eur": 200.0},
        {"id": "INC-2", "service": "payments-service", "root_cause_category": "database_issue",
         "incident_cost_eur": 500.0, "toil_cost_eur": 100.0},
    ])


def _risk_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"service": "payments-service", "incidents_per_month": 10.0, "mttr_p90_minutes": 120.0, "change_failure_rate": 0.20},
        {"service": "search-service", "incidents_per_month": 2.0, "mttr_p90_minutes": 30.0, "change_failure_rate": 0.05},
    ])


# ---------------- whatif_priced_incidents ----------------

def test_whatif_priced_incidents_no_reduction_is_identity():
    df = whatif_priced_incidents(_priced_incidents(), mttr_reduction_pct=0, cfr_reduction_pct=0)
    assert df["incident_cost_eur"].tolist() == [1000.0, 500.0]
    assert df["toil_cost_eur"].tolist() == [200.0, 100.0]


def test_whatif_priced_incidents_full_mttr_reduction_zeroes_cost():
    df = whatif_priced_incidents(_priced_incidents(), mttr_reduction_pct=100, cfr_reduction_pct=0)
    assert df["incident_cost_eur"].tolist() == [0.0, 0.0]
    assert df["toil_cost_eur"].tolist() == [0.0, 0.0]


def test_whatif_priced_incidents_cfr_only_affects_deploy_caused_category():
    df = whatif_priced_incidents(_priced_incidents(), mttr_reduction_pct=0, cfr_reduction_pct=50)
    row1 = df[df["id"] == "INC-1"].iloc[0]  # deployment_regression -> affected
    row2 = df[df["id"] == "INC-2"].iloc[0]  # database_issue -> unaffected
    assert row1["incident_cost_eur"] == pytest.approx(500.0)
    assert row2["incident_cost_eur"] == pytest.approx(500.0)  # unchanged


def test_whatif_priced_incidents_combined_reductions_stack():
    df = whatif_priced_incidents(_priced_incidents(), mttr_reduction_pct=50, cfr_reduction_pct=50)
    row1 = df[df["id"] == "INC-1"].iloc[0]  # both reductions apply: 1000 * 0.5 * 0.5
    row2 = df[df["id"] == "INC-2"].iloc[0]  # only mttr reduction: 500 * 0.5
    assert row1["incident_cost_eur"] == pytest.approx(250.0)
    assert row2["incident_cost_eur"] == pytest.approx(250.0)


# ---------------- whatif_risk_scores ----------------

def test_whatif_risk_scores_no_reduction_matches_baseline_ordering():
    df = whatif_risk_scores(_risk_df(), mttr_reduction_pct=0, cfr_reduction_pct=0)
    assert df.iloc[0]["service"] == "payments-service"  # higher on every raw signal


def test_whatif_risk_scores_empty_input():
    df = whatif_risk_scores(pd.DataFrame(columns=["service", "incidents_per_month", "mttr_p90_minutes", "change_failure_rate"]),
                             mttr_reduction_pct=10, cfr_reduction_pct=10)
    assert len(df) == 0
    assert "whatif_risk_score" in df.columns


def test_whatif_risk_scores_large_reduction_narrows_the_gap():
    baseline = whatif_risk_scores(_risk_df(), mttr_reduction_pct=0, cfr_reduction_pct=0)
    improved = whatif_risk_scores(_risk_df(), mttr_reduction_pct=90, cfr_reduction_pct=90)
    baseline_gap = baseline.iloc[0]["whatif_risk_score"] - baseline.iloc[1]["whatif_risk_score"]
    improved_gap = improved.iloc[0]["whatif_risk_score"] - improved.iloc[1]["whatif_risk_score"]
    # payments-service's mttr/cfr signals shrink toward search-service's -- incident
    # frequency (untouched by these two sliders) still separates them, but less starkly
    assert improved_gap <= baseline_gap
