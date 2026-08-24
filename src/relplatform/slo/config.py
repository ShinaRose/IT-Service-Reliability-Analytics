"""Loads and validates config/slo.yaml -- per-service SLO targets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from relplatform.config import ROOT

SLO_CONFIG_PATH = ROOT / "config" / "slo.yaml"


@dataclass(frozen=True)
class SLOTarget:
    service: str
    availability_target_pct: float
    latency_target_ms: float
    measurement_window_days: float

    def __post_init__(self):
        if not (0 < self.availability_target_pct <= 100):
            raise ValueError(f"{self.service}: availability_target_pct must be in (0, 100], got {self.availability_target_pct}")
        if self.latency_target_ms <= 0:
            raise ValueError(f"{self.service}: latency_target_ms must be > 0, got {self.latency_target_ms}")
        if self.measurement_window_days <= 0:
            raise ValueError(f"{self.service}: measurement_window_days must be > 0, got {self.measurement_window_days}")


def load_slo_config(path: Path | str = SLO_CONFIG_PATH) -> dict[str, SLOTarget]:
    """Returns {service: SLOTarget} for every service listed under `services:`, each
    merged over `default:` (a service section only needs to override what it changes)."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    default = raw.get("default", {})
    services_raw = raw.get("services", {})
    if not services_raw:
        raise ValueError(f"{path}: no services defined under 'services:'")

    targets = {}
    for service, overrides in services_raw.items():
        merged = {**default, **(overrides or {})}
        missing = {"availability_target_pct", "latency_target_ms", "measurement_window_days"} - merged.keys()
        if missing:
            raise ValueError(f"{path}: service '{service}' missing required field(s) {missing} (not in service section or default)")
        targets[service] = SLOTarget(
            service=service,
            availability_target_pct=float(merged["availability_target_pct"]),
            latency_target_ms=float(merged["latency_target_ms"]),
            measurement_window_days=float(merged["measurement_window_days"]),
        )
    return targets
