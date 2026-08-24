from datetime import datetime, timedelta

import pandas as pd
import pytest

from relplatform.finance.config import CostConfig, load_cost_config
from relplatform.finance.counterfactual import change_failure_rate_uplift, time_to_restore_uplift
from relplatform.finance.incident_cost import incident_cost_eur, incident_costs
from relplatform.finance.rerank import euro_impact_by_service, side_by_side_ranking
from relplatform.finance.toil_cost import toil_cost_eur, toil_costs, toil_by_root_cause, toil_by_service

T0 = datetime(2026, 1, 1, 0, 0, 0)


def _cost_config(**overrides) -> CostConfig:
    base = dict(
        loaded_hourly_rate_eur=90.0,
        affected_user_fraction_by_severity={"SEV1": 1.0, "SEV2": 0.5, "SEV3": 0.2, "SEV4": 0.05},
        responders_by_severity={"SEV1": 3, "SEV2": 2, "SEV3": 1, "SEV4": 1},
        default_downtime_cost_eur_per_minute=50.0,
        downtime_cost_eur_per_minute={"payments-service": 400.0},
    )
    base.update(overrides)
    return CostConfig(**base)


def _incidents(rows: list[dict]) -> pd.DataFrame:
    cols = ["id", "service", "severity", "started_at", "acknowledged_at", "resolved_at", "root_cause_category"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


# ---------------- config ----------------

def test_load_real_costs_config():
    cfg = load_cost_config()
    assert cfg.loaded_hourly_rate_eur > 0
    assert cfg.downtime_rate("payments-service") > cfg.downtime_rate("notification-service")


def test_cost_config_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        _validate_fraction_out_of_range()


def _validate_fraction_out_of_range():
    from relplatform.finance.config import load_cost_config
    import tempfile, yaml, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({
            "engineering": {"loaded_hourly_rate_eur": 90},
            "affected_user_fraction_by_severity": {"SEV1": 1.5},  # invalid, > 1
        }, f)
        path = f.name
    try:
        load_cost_config(path)
    finally:
        os.unlink(path)


def test_downtime_rate_falls_back_to_default():
    cfg = _cost_config()
    assert cfg.downtime_rate("unknown-service") == 50.0


# ---------------- incident cost ----------------

def test_incident_cost_basic():
    cfg = _cost_config()
    # 60 minutes * 400 eur/min * 1.0 (SEV1) = 24000
    assert incident_cost_eur(60, "payments-service", "SEV1", cfg) == pytest.approx(24000.0)


def test_incident_cost_zero_duration():
    cfg = _cost_config()
    assert incident_cost_eur(0, "payments-service", "SEV1", cfg) == 0.0


def test_incident_cost_negative_duration_clips_to_zero():
    cfg = _cost_config()
    assert incident_cost_eur(-10, "payments-service", "SEV1", cfg) == 0.0


def test_incident_costs_empty_dataframe():
    cfg = _cost_config()
    result = incident_costs(_incidents([]), cfg)
    assert len(result) == 0
    assert "incident_cost_eur" in result.columns


def test_incident_costs_single_row():
    cfg = _cost_config()
    df = _incidents([{
        "id": "INC-1", "service": "payments-service", "severity": "SEV2",
        "started_at": T0, "acknowledged_at": T0 + timedelta(minutes=5), "resolved_at": T0 + timedelta(minutes=35),
        "root_cause_category": "deployment_regression",
    }])
    result = incident_costs(df, cfg)
    # duration is started_at -> resolved_at (35 min), not acknowledged_at -> resolved_at
    assert result.iloc[0]["duration_minutes"] == pytest.approx(35)
    assert result.iloc[0]["incident_cost_eur"] == pytest.approx(35 * 400 * 0.5)


# ---------------- toil cost ----------------

def test_toil_cost_basic():
    cfg = _cost_config()
    # 60 response minutes, SEV1 -> 3 responders -> 3 hours toil * 90 eur/hr = 270
    assert toil_cost_eur(60, "SEV1", cfg) == pytest.approx(270.0)


def test_toil_cost_zero_response_time():
    cfg = _cost_config()
    assert toil_cost_eur(0, "SEV1", cfg) == 0.0


def test_toil_costs_acknowledged_after_resolved_clips_to_zero():
    # Defensive: shouldn't happen in real data, but must not go negative.
    cfg = _cost_config()
    df = _incidents([{
        "id": "INC-1", "service": "payments-service", "severity": "SEV1",
        "started_at": T0, "acknowledged_at": T0 + timedelta(minutes=20), "resolved_at": T0 + timedelta(minutes=10),
        "root_cause_category": "deployment_regression",
    }])
    result = toil_costs(df, cfg)
    assert result.iloc[0]["response_minutes"] == 0
    assert result.iloc[0]["toil_cost_eur"] == 0


def test_toil_by_service_and_root_cause_grouping():
    cfg = _cost_config()
    df = _incidents([
        {"id": "INC-1", "service": "payments-service", "severity": "SEV1", "started_at": T0,
         "acknowledged_at": T0 + timedelta(minutes=5), "resolved_at": T0 + timedelta(minutes=65),
         "root_cause_category": "deployment_regression"},
        {"id": "INC-2", "service": "payments-service", "severity": "SEV3", "started_at": T0,
         "acknowledged_at": T0 + timedelta(minutes=5), "resolved_at": T0 + timedelta(minutes=25),
         "root_cause_category": "database_issue"},
    ])
    with_toil = toil_costs(df, cfg)
    by_svc = toil_by_service(with_toil)
    by_cause = toil_by_root_cause(with_toil)
    assert len(by_svc) == 1
    assert by_svc.iloc[0]["n_incidents"] == 2
    assert len(by_cause) == 2


# ---------------- counterfactual: time to restore ----------------

def test_ttr_uplift_no_incidents():
    cfg = _cost_config()
    result = time_to_restore_uplift(_incidents([]), "payments-service", cfg)
    assert result.status == "insufficient_data"
    assert result.euros_saved_per_year is None


def test_ttr_uplift_already_elite():
    cfg = _cost_config()
    df = _incidents([{
        "id": "INC-1", "service": "payments-service", "severity": "SEV4", "started_at": T0,
        "acknowledged_at": T0 + timedelta(minutes=1), "resolved_at": T0 + timedelta(minutes=5),
        "root_cause_category": "deployment_regression",
    }])
    df = incident_costs(df, cfg)
    df = toil_costs(df, cfg)
    result = time_to_restore_uplift(df, "payments-service", cfg)
    assert result.current_band == "elite"
    assert result.status == "already_best_band"
    assert result.euros_saved_per_year is None


def test_ttr_uplift_modeled_reduces_but_never_negative():
    cfg = _cost_config()
    rows = [{
        "id": f"INC-{i}", "service": "payments-service", "severity": "SEV2", "started_at": T0,
        "acknowledged_at": T0 + timedelta(minutes=10), "resolved_at": T0 + timedelta(hours=48),  # 2 days -> "medium" band
        "root_cause_category": "database_issue",
    } for i in range(5)]
    df = incident_costs(_incidents(rows), cfg)
    df = toil_costs(df, cfg)
    result = time_to_restore_uplift(df, "payments-service", cfg)
    assert result.status == "modeled"
    assert result.current_band == "medium"
    assert result.target_band == "high"
    assert result.hours_saved_per_year > 0
    assert result.euros_saved_per_year > 0


# ---------------- counterfactual: change failure rate ----------------

def test_cfr_uplift_no_deploys():
    cfg = _cost_config()
    result = change_failure_rate_uplift("payments-service", 0, 20.0, _incidents([]), cfg)
    assert result.status == "insufficient_data"


def test_cfr_uplift_already_elite():
    cfg = _cost_config()
    result = change_failure_rate_uplift("payments-service", 500, 5.0, _incidents([]), cfg)
    assert result.status == "already_best_band"
    assert result.current_band == "elite"


def test_cfr_uplift_no_historical_deploy_caused_incidents():
    cfg = _cost_config()
    result = change_failure_rate_uplift("payments-service", 500, 20.0, _incidents([]), cfg)
    assert result.status == "insufficient_data"
    assert result.euros_saved_per_year is None


def test_cfr_uplift_modeled():
    cfg = _cost_config()
    rows = [{
        "id": f"INC-{i}", "service": "payments-service", "severity": "SEV2", "started_at": T0,
        "acknowledged_at": T0 + timedelta(minutes=10), "resolved_at": T0 + timedelta(minutes=70),
        "root_cause_category": "deployment_regression",
    } for i in range(3)]
    df = incident_costs(_incidents(rows), cfg)
    df = toil_costs(df, cfg)
    result = change_failure_rate_uplift("payments-service", deploys_per_year=1000, current_rate_pct=20.0,
                                         deploy_caused_incidents=df, cost_config=cfg)
    assert result.status == "modeled"
    assert result.current_band == "high"
    assert result.target_band == "elite"
    # (1000 deploys/yr) * (20% - 15%) = 50 incidents avoided/year, each averaging
    # (70 min duration, started_at -> resolved_at) = 1.1667h
    assert result.hours_saved_per_year == pytest.approx(50 * (70 / 60), rel=0.02)
    assert result.euros_saved_per_year > 0


# ---------------- rerank ----------------

def test_euro_impact_by_service_empty():
    cfg = _cost_config()
    df = incident_costs(_incidents([]), cfg)
    df = toil_costs(df, cfg)
    result = euro_impact_by_service(df)
    assert len(result) == 0


def test_side_by_side_ranking_single_service():
    risk_df = pd.DataFrame([{"service": "payments-service", "risk_score": 50.0}])
    euro_df = pd.DataFrame([{"service": "payments-service", "total_cost_eur": 1000.0}])
    result = side_by_side_ranking(risk_df, euro_df)
    assert len(result) == 1
    assert result.iloc[0]["rank_delta"] == 0


def test_side_by_side_ranking_shows_divergence():
    # payments is #2 by risk_score but #1 by euro cost -- rank_delta should show it moved up.
    risk_df = pd.DataFrame([
        {"service": "checkout-service", "risk_score": 80.0},
        {"service": "payments-service", "risk_score": 60.0},
    ])
    euro_df = pd.DataFrame([
        {"service": "checkout-service", "total_cost_eur": 500.0},
        {"service": "payments-service", "total_cost_eur": 5000.0},
    ])
    result = side_by_side_ranking(risk_df, euro_df)
    payments_row = result[result["service"] == "payments-service"].iloc[0]
    assert payments_row["risk_rank"] == 2
    assert payments_row["euro_rank"] == 1
    assert payments_row["rank_delta"] == 1  # moved up one place under the euro ranking
