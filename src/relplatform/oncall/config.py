"""Loads and validates config/oncall.yaml."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from relplatform.config import ROOT

ONCALL_CONFIG_PATH = ROOT / "config" / "oncall.yaml"


@dataclass(frozen=True)
class OnCallConfig:
    business_start_hour: int
    business_end_hour: int
    business_weekdays: tuple[int, ...]
    sleep_start_hour: int
    sleep_end_hour: int

    def is_business_hours(self, ts) -> bool:
        return ts.weekday() in self.business_weekdays and self.business_start_hour <= ts.hour < self.business_end_hour

    def is_sleep_hours(self, ts) -> bool:
        return _in_wrapping_window(ts.hour, self.sleep_start_hour, self.sleep_end_hour)


def _in_wrapping_window(hour: int, start: int, end: int) -> bool:
    """True if `hour` falls in [start, end), where the window may wrap past midnight
    (start > end, e.g. 23 -> 7)."""
    if start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def load_oncall_config(path: Path | str = ONCALL_CONFIG_PATH) -> OnCallConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    bh = raw.get("business_hours", {})
    sh = raw.get("sleep_hours", {})

    for label, block in (("business_hours", bh), ("sleep_hours", sh)):
        for key in ("start_hour", "end_hour"):
            if key not in block:
                raise ValueError(f"{path}: {label}.{key} is required")
            if not (0 <= int(block[key]) <= 24):
                raise ValueError(f"{path}: {label}.{key} must be in [0,24], got {block[key]}")

    weekdays = tuple(int(d) for d in bh.get("weekdays", [0, 1, 2, 3, 4]))
    for d in weekdays:
        if not (0 <= d <= 6):
            raise ValueError(f"{path}: business_hours.weekdays entries must be in [0,6], got {d}")

    return OnCallConfig(
        business_start_hour=int(bh["start_hour"]),
        business_end_hour=int(bh["end_hour"]),
        business_weekdays=weekdays,
        sleep_start_hour=int(sh["start_hour"]),
        sleep_end_hour=int(sh["end_hour"]),
    )
