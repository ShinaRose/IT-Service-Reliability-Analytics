from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from relplatform.oncall.config import OnCallConfig, load_oncall_config
from relplatform.oncall.fatigue import alert_fatigue_by_service, noise_ratio_by_service
from relplatform.oncall.pages import (
    assign_pages_to_shifts,
    gini_coefficient,
    interrupt_concentration,
    out_of_hours_rate,
    pages_per_shift,
    pages_per_shift_percentiles,
    sleep_hours_interruptions,
)

T0 = datetime(2026, 1, 5, 0, 0, 0)  # a Monday


def _cfg(**overrides) -> OnCallConfig:
    base = dict(
        business_start_hour=9, business_end_hour=18, business_weekdays=(0, 1, 2, 3, 4),
        sleep_start_hour=23, sleep_end_hour=7,
    )
    base.update(overrides)
    return OnCallConfig(**base)


def _shifts(rows: list[dict]) -> pd.DataFrame:
    cols = ["id", "engineer", "shift_start", "shift_end", "is_holiday", "swapped"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _incidents(rows: list[dict]) -> pd.DataFrame:
    cols = ["id", "service", "started_at"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# ---------------- config ----------------

def test_load_real_oncall_config():
    cfg = load_oncall_config()
    assert cfg.business_start_hour < cfg.business_end_hour
    assert 0 <= cfg.sleep_start_hour <= 24


def test_config_rejects_invalid_hour():
    import os
    import tempfile

    import yaml
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"business_hours": {"start_hour": 30, "end_hour": 18}, "sleep_hours": {"start_hour": 23, "end_hour": 7}}, f)
        path = f.name
    try:
        with pytest.raises(ValueError):
            load_oncall_config(path)
    finally:
        os.unlink(path)


def test_is_business_hours_weekday_and_weekend():
    cfg = _cfg()
    assert cfg.is_business_hours(datetime(2026, 1, 5, 10, 0)) is True   # Monday 10:00
    assert cfg.is_business_hours(datetime(2026, 1, 5, 8, 0)) is False   # Monday 08:00, before open
    assert cfg.is_business_hours(datetime(2026, 1, 5, 18, 0)) is False  # Monday 18:00, end is exclusive
    assert cfg.is_business_hours(datetime(2026, 1, 10, 10, 0)) is False  # Saturday


def test_is_sleep_hours_wraps_midnight():
    cfg = _cfg()
    assert cfg.is_sleep_hours(datetime(2026, 1, 5, 23, 0)) is True
    assert cfg.is_sleep_hours(datetime(2026, 1, 5, 0, 0)) is True
    assert cfg.is_sleep_hours(datetime(2026, 1, 5, 6, 59)) is True
    assert cfg.is_sleep_hours(datetime(2026, 1, 5, 7, 0)) is False  # end exclusive
    assert cfg.is_sleep_hours(datetime(2026, 1, 5, 12, 0)) is False


# ---------------- assign_pages_to_shifts ----------------

def test_assign_pages_empty_incidents():
    result = assign_pages_to_shifts(_incidents([]), _shifts([{
        "id": "SHIFT-0001", "engineer": "oncall-eng-1", "shift_start": T0, "shift_end": T0 + timedelta(days=7),
        "is_holiday": False, "swapped": False,
    }]))
    assert len(result) == 0
    assert "engineer" in result.columns


def test_assign_pages_empty_shifts():
    result = assign_pages_to_shifts(_incidents([
        {"id": "INC-1", "service": "payments-service", "started_at": T0 + timedelta(hours=2)},
    ]), _shifts([]))
    assert len(result) == 1
    assert result.iloc[0]["engineer"] is None


def test_assign_pages_basic():
    shifts = _shifts([
        {"id": "SHIFT-0001", "engineer": "oncall-eng-1", "shift_start": T0, "shift_end": T0 + timedelta(days=7),
         "is_holiday": False, "swapped": False},
        {"id": "SHIFT-0002", "engineer": "oncall-eng-2", "shift_start": T0 + timedelta(days=7), "shift_end": T0 + timedelta(days=14),
         "is_holiday": False, "swapped": False},
    ])
    incidents = _incidents([
        {"id": "INC-1", "service": "payments-service", "started_at": T0 + timedelta(days=1)},
        {"id": "INC-2", "service": "payments-service", "started_at": T0 + timedelta(days=8)},
    ])
    result = assign_pages_to_shifts(incidents, shifts)
    assert result.set_index("id").loc["INC-1", "engineer"] == "oncall-eng-1"
    assert result.set_index("id").loc["INC-2", "engineer"] == "oncall-eng-2"
    # incident's own id column must survive the merge, not get shadowed by the shift's id
    assert set(result["id"]) == {"INC-1", "INC-2"}


# ---------------- pages per shift ----------------

def test_pages_per_shift_percentiles_no_shifts():
    result = pages_per_shift_percentiles(_incidents([]).assign(engineer=[], shift_id=[]), _shifts([]))
    assert result["n_shifts"] == 0
    assert result["percentiles"] == {}


def test_pages_per_shift_percentiles_includes_zero_page_shifts():
    shifts = _shifts([
        {"id": "SHIFT-0001", "engineer": "oncall-eng-1", "shift_start": T0, "shift_end": T0 + timedelta(days=7),
         "is_holiday": False, "swapped": False},
        {"id": "SHIFT-0002", "engineer": "oncall-eng-2", "shift_start": T0 + timedelta(days=7), "shift_end": T0 + timedelta(days=14),
         "is_holiday": False, "swapped": False},
    ])
    incidents = _incidents([{"id": "INC-1", "service": "payments-service", "started_at": T0 + timedelta(days=1)}])
    paged = assign_pages_to_shifts(incidents, shifts)
    result = pages_per_shift_percentiles(paged, shifts)
    assert result["n_shifts"] == 2
    # one shift paged once, one shift paged zero times -> mean is 0.5, not 1
    assert result["mean"] == pytest.approx(0.5)


def test_pages_per_shift_single_shift():
    shifts = _shifts([{"id": "SHIFT-0001", "engineer": "oncall-eng-1", "shift_start": T0, "shift_end": T0 + timedelta(days=7),
                        "is_holiday": False, "swapped": False}])
    incidents = _incidents([
        {"id": f"INC-{i}", "service": "payments-service", "started_at": T0 + timedelta(hours=i)} for i in range(3)
    ])
    paged = assign_pages_to_shifts(incidents, shifts)
    by_shift = pages_per_shift(paged)
    assert len(by_shift) == 1
    assert by_shift.iloc[0]["n_pages"] == 3


# ---------------- out-of-hours / sleep-hours ----------------

def test_out_of_hours_rate_empty():
    result = out_of_hours_rate(_incidents([]).assign(engineer=[]), _cfg())
    assert result["out_of_hours_rate"] is None


def test_out_of_hours_rate_basic():
    cfg = _cfg()
    df = pd.DataFrame([
        {"engineer": "oncall-eng-1", "started_at": datetime(2026, 1, 5, 10, 0)},  # business hours
        {"engineer": "oncall-eng-1", "started_at": datetime(2026, 1, 5, 22, 0)},  # out of hours
        {"engineer": "oncall-eng-1", "started_at": datetime(2026, 1, 10, 10, 0)},  # weekend -> out of hours
    ])
    result = out_of_hours_rate(df, cfg)
    assert result["n_pages"] == 3
    assert result["n_out_of_hours"] == 2
    assert result["out_of_hours_rate"] == pytest.approx(2 / 3)


def test_sleep_hours_interruptions_basic():
    cfg = _cfg()
    df = pd.DataFrame([
        {"engineer": "oncall-eng-1", "started_at": datetime(2026, 1, 5, 3, 0)},   # sleep hours
        {"engineer": "oncall-eng-1", "started_at": datetime(2026, 1, 5, 14, 0)},  # not
    ])
    result = sleep_hours_interruptions(df, cfg)
    assert result["n_sleep_hours"] == 1
    assert result["sleep_hours_rate"] == pytest.approx(0.5)


# ---------------- concentration ----------------

def test_gini_coefficient_even_split_is_zero():
    assert gini_coefficient(np.array([5, 5, 5, 5])) == pytest.approx(0.0)


def test_gini_coefficient_empty_is_zero():
    assert gini_coefficient(np.array([])) == 0.0


def test_gini_coefficient_concentrated_load():
    # one engineer took all 10 pages, three took none
    assert gini_coefficient(np.array([0, 0, 0, 10])) == pytest.approx(0.75)


def test_interrupt_concentration_empty():
    result = interrupt_concentration(_incidents([]).assign(engineer=[]))
    assert result["n_engineers"] == 0
    assert result["gini"] == 0.0


def test_interrupt_concentration_single_engineer():
    df = pd.DataFrame([{"engineer": "oncall-eng-1"} for _ in range(5)])
    result = interrupt_concentration(df)
    assert result["n_engineers"] == 1
    assert result["top1_share"] == 1.0


# ---------------- alert fatigue ----------------

def test_noise_ratio_by_service_empty():
    result = noise_ratio_by_service(pd.DataFrame(columns=["service", "cluster_id"]))
    assert len(result) == 0


def test_alert_fatigue_by_service_basic():
    clustered = pd.DataFrame([
        {"service": "payments-service", "cluster_id": "a"},
        {"service": "payments-service", "cluster_id": "a"},
        {"service": "payments-service", "cluster_id": "b"},
        {"service": "search-service", "cluster_id": "c"},
    ])
    paged = pd.DataFrame([
        {"service": "payments-service", "engineer": "oncall-eng-1"},
        {"service": "payments-service", "engineer": "oncall-eng-1"},
    ])
    result = alert_fatigue_by_service(clustered, paged, months=1.0)
    assert len(result) == 2
    payments_row = result[result["service"] == "payments-service"].iloc[0]
    search_row = result[result["service"] == "search-service"].iloc[0]
    # payments has both more noise (3 alerts / 2 clusters) and more pages -> higher fatigue
    assert payments_row["fatigue_score"] > search_row["fatigue_score"]
    assert search_row["n_pages"] == 0


def test_alert_fatigue_by_service_empty_clustered():
    result = alert_fatigue_by_service(pd.DataFrame(columns=["service", "cluster_id"]), pd.DataFrame(columns=["service", "engineer"]), months=1.0)
    assert len(result) == 0
