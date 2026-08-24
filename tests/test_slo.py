from datetime import datetime, timedelta

import pandas as pd
import pytest

from relplatform.slo.budget import compute_error_budget, downtime_minutes_in_window
from relplatform.slo.burn_rate import BURN_RATE_RULES, evaluate_burn_rate_alerts, project_exhaustion_date
from relplatform.slo.config import SLOTarget, load_slo_config
from relplatform.slo.freeze import CHANGE_FAILURE_RISK_THRESHOLD_PCT, recommend

AS_OF = datetime(2026, 8, 24, 12, 0, 0)


def _target(**overrides) -> SLOTarget:
    base = dict(service="test-service", availability_target_pct=99.9, latency_target_ms=300, measurement_window_days=30)
    base.update(overrides)
    return SLOTarget(**base)


def _incidents(rows: list[tuple[datetime, datetime]], service: str = "test-service") -> pd.DataFrame:
    return pd.DataFrame([
        {"service": service, "started_at": s, "resolved_at": e} for s, e in rows
    ]) if rows else pd.DataFrame(columns=["service", "started_at", "resolved_at"])


# ---------------- config ----------------

def test_load_real_slo_config():
    targets = load_slo_config()
    assert "payments-service" in targets
    assert targets["payments-service"].availability_target_pct == 98.5


def test_slo_target_rejects_invalid_values():
    with pytest.raises(ValueError):
        _target(availability_target_pct=0)
    with pytest.raises(ValueError):
        _target(availability_target_pct=101)
    with pytest.raises(ValueError):
        _target(latency_target_ms=0)
    with pytest.raises(ValueError):
        _target(measurement_window_days=0)


# ---------------- budget: boundary conditions ----------------

def test_no_incidents_in_window():
    incidents = _incidents([])
    status = compute_error_budget(incidents, _target(), AS_OF)
    assert status.downtime_minutes == 0
    assert status.budget_consumed_pct == 0.0
    assert status.exhausted is False
    assert status.budget_remaining_minutes == pytest.approx(status.error_budget_minutes)


def test_zero_budget_target_100_pct_no_downtime():
    target = _target(availability_target_pct=100.0)
    incidents = _incidents([])
    status = compute_error_budget(incidents, target, AS_OF)
    assert status.error_budget_minutes == 0
    assert status.budget_consumed_pct == 0.0
    assert status.exhausted is False


def test_zero_budget_target_100_pct_with_downtime_is_exhausted_not_a_crash():
    target = _target(availability_target_pct=100.0)
    incidents = _incidents([(AS_OF - timedelta(hours=2), AS_OF - timedelta(hours=1))])
    status = compute_error_budget(incidents, target, AS_OF)
    assert status.error_budget_minutes == 0
    assert status.budget_consumed_pct == float("inf")
    assert status.exhausted is True
    assert status.budget_remaining_minutes < 0


def test_budget_already_exhausted():
    # 99% target over 30 days -> ~432 min budget; one 8-hour incident (480 min) exceeds it.
    target = _target(availability_target_pct=99.0)
    incidents = _incidents([(AS_OF - timedelta(hours=8), AS_OF)])
    status = compute_error_budget(incidents, target, AS_OF)
    assert status.exhausted is True
    assert status.budget_remaining_minutes < 0
    assert status.budget_consumed_pct > 100


def test_downtime_clips_to_window_boundary():
    target = _target(measurement_window_days=1)  # 24h window
    # incident starts 2 days ago, ends 12h ago -- only the last 12h of it fall in a 24h window
    incidents = _incidents([(AS_OF - timedelta(days=2), AS_OF - timedelta(hours=12))])
    downtime = downtime_minutes_in_window(incidents, "test-service", AS_OF, window_hours=24)
    assert downtime == pytest.approx(12 * 60)


# ---------------- burn rate ----------------

def test_burn_rate_thresholds_match_google_sre_workbook_30_day_period():
    target = _target(measurement_window_days=30)
    incidents = _incidents([])
    alerts = evaluate_burn_rate_alerts(incidents, target, AS_OF)
    fast = next(a for a in alerts if a.rule_name == "fast_burn")
    slow = next(a for a in alerts if a.rule_name == "slow_burn")
    assert fast.threshold == pytest.approx(14.4, abs=0.01)
    assert slow.threshold == pytest.approx(6.0, abs=0.01)


def test_burn_rate_not_firing_with_no_incidents():
    target = _target()
    incidents = _incidents([])
    alerts = evaluate_burn_rate_alerts(incidents, target, AS_OF)
    assert all(not a.firing for a in alerts)


def test_fast_burn_fires_on_sustained_recent_outage():
    # 99.9% / 30 days: allowed bad fraction = 0.001. A total outage for the last hour
    # (both the 1h long window and 5min short window are 100% down) has burn rate
    # 1.0/0.001 = 1000, far above the 14.4 threshold, and it's sustained across both windows.
    target = _target(availability_target_pct=99.9, measurement_window_days=30)
    incidents = _incidents([(AS_OF - timedelta(hours=1), AS_OF)])
    alerts = evaluate_burn_rate_alerts(incidents, target, AS_OF)
    fast = next(a for a in alerts if a.rule_name == "fast_burn")
    assert fast.firing is True
    assert fast.long_window_burn_rate > fast.threshold
    assert fast.short_window_burn_rate > fast.threshold


def test_burn_rate_requires_both_windows_short_window_already_recovered():
    # Outage covered the long window (1h) but resolved 10 minutes ago -- the short
    # window (5 min) is now clean, so multi-window alerting should NOT fire even though
    # the long window alone still looks bad. This is the "fast reset" property.
    target = _target(availability_target_pct=99.9, measurement_window_days=30)
    incidents = _incidents([(AS_OF - timedelta(hours=1), AS_OF - timedelta(minutes=10))])
    alerts = evaluate_burn_rate_alerts(incidents, target, AS_OF)
    fast = next(a for a in alerts if a.rule_name == "fast_burn")
    assert fast.short_window_burn_rate == 0 or fast.short_window_burn_rate is None
    assert fast.firing is False


def test_burn_rate_rules_are_exactly_fast_and_slow():
    names = {name for name, _, _ in BURN_RATE_RULES}
    assert names == {"fast_burn", "slow_burn"}


# ---------------- exhaustion projection ----------------

def test_exhaustion_projection_already_exhausted():
    target = _target()
    incidents = _incidents([(AS_OF - timedelta(hours=1), AS_OF)])
    proj = project_exhaustion_date(incidents, target, AS_OF, remaining_minutes=-10)
    assert proj.status == "already_exhausted"
    assert proj.central_date is None


def test_exhaustion_projection_no_recent_burn():
    target = _target()
    incidents = _incidents([])
    proj = project_exhaustion_date(incidents, target, AS_OF, remaining_minutes=100)
    assert proj.status == "no_recent_burn"


def test_exhaustion_projection_with_recent_burn():
    target = _target(availability_target_pct=99.0, measurement_window_days=30)
    incidents = _incidents([(AS_OF - timedelta(hours=2), AS_OF - timedelta(hours=1))])
    proj = project_exhaustion_date(incidents, target, AS_OF, remaining_minutes=200)
    assert proj.status == "projected"
    assert proj.central_date is not None
    assert proj.optimistic_date is not None
    assert proj.pessimistic_date is not None


# ---------------- freeze recommendation + change-failure cross-check ----------------

def test_freeze_green_when_healthy():
    target = _target()
    incidents = _incidents([])
    budget = compute_error_budget(incidents, target, AS_OF)
    alerts = evaluate_burn_rate_alerts(incidents, target, AS_OF)
    rec = recommend(budget, alerts, change_failure_rate_pct=5.0)
    assert rec.light == "green"
    assert rec.disagreement is False


def test_freeze_red_when_exhausted():
    target = _target(availability_target_pct=99.0)
    incidents = _incidents([(AS_OF - timedelta(hours=8), AS_OF)])
    budget = compute_error_budget(incidents, target, AS_OF)
    alerts = evaluate_burn_rate_alerts(incidents, target, AS_OF)
    rec = recommend(budget, alerts, change_failure_rate_pct=5.0)
    assert rec.light == "red"


def test_freeze_disagreement_surfaced_when_slo_red_but_model_says_low_risk():
    target = _target(availability_target_pct=99.0)
    incidents = _incidents([(AS_OF - timedelta(hours=8), AS_OF)])
    budget = compute_error_budget(incidents, target, AS_OF)
    alerts = evaluate_burn_rate_alerts(incidents, target, AS_OF)
    rec = recommend(budget, alerts, change_failure_rate_pct=2.0)  # well under the 15% threshold
    assert rec.light == "red"
    assert rec.disagreement is True
    assert "low-risk" in rec.disagreement_note


def test_freeze_disagreement_surfaced_when_slo_green_but_model_says_risky():
    target = _target()
    incidents = _incidents([])
    budget = compute_error_budget(incidents, target, AS_OF)
    alerts = evaluate_burn_rate_alerts(incidents, target, AS_OF)
    rec = recommend(budget, alerts, change_failure_rate_pct=CHANGE_FAILURE_RISK_THRESHOLD_PCT + 5)
    assert rec.light == "green"
    assert rec.disagreement is True
    assert "risky" in rec.disagreement_note


def test_freeze_no_disagreement_when_change_failure_rate_unavailable():
    target = _target()
    incidents = _incidents([])
    budget = compute_error_budget(incidents, target, AS_OF)
    alerts = evaluate_burn_rate_alerts(incidents, target, AS_OF)
    rec = recommend(budget, alerts, change_failure_rate_pct=None)
    assert rec.disagreement is False
    assert rec.change_failure_model_flags_risky is None
