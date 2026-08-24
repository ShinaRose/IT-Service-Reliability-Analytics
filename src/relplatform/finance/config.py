"""Loads and validates config/costs.yaml."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from relplatform.config import ROOT

COSTS_CONFIG_PATH = ROOT / "config" / "costs.yaml"

SEVERITIES = ["SEV1", "SEV2", "SEV3", "SEV4"]


@dataclass(frozen=True)
class CostConfig:
    loaded_hourly_rate_eur: float
    affected_user_fraction_by_severity: dict[str, float]
    responders_by_severity: dict[str, float]
    default_downtime_cost_eur_per_minute: float
    downtime_cost_eur_per_minute: dict[str, float]

    def downtime_rate(self, service: str) -> float:
        return self.downtime_cost_eur_per_minute.get(service, self.default_downtime_cost_eur_per_minute)

    def affected_fraction(self, severity: str) -> float:
        return self.affected_user_fraction_by_severity.get(severity, 0.1)

    def responders(self, severity: str) -> float:
        return self.responders_by_severity.get(severity, 1)


def load_cost_config(path: Path | str = COSTS_CONFIG_PATH) -> CostConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    eng = raw.get("engineering", {})
    if "loaded_hourly_rate_eur" not in eng:
        raise ValueError(f"{path}: engineering.loaded_hourly_rate_eur is required")
    rate = float(eng["loaded_hourly_rate_eur"])
    if rate <= 0:
        raise ValueError(f"{path}: engineering.loaded_hourly_rate_eur must be > 0, got {rate}")

    fractions = {k: float(v) for k, v in raw.get("affected_user_fraction_by_severity", {}).items()}
    for sev, frac in fractions.items():
        if not (0 <= frac <= 1):
            raise ValueError(f"{path}: affected_user_fraction_by_severity[{sev}] must be in [0,1], got {frac}")

    responders = {k: float(v) for k, v in raw.get("responders_by_severity", {}).items()}

    default_rate = float(raw.get("default_downtime_cost_eur_per_minute", 0))
    if default_rate < 0:
        raise ValueError(f"{path}: default_downtime_cost_eur_per_minute must be >= 0")

    per_service = {k: float(v) for k, v in raw.get("downtime_cost_eur_per_minute", {}).items()}

    return CostConfig(
        loaded_hourly_rate_eur=rate,
        affected_user_fraction_by_severity=fractions,
        responders_by_severity=responders,
        default_downtime_cost_eur_per_minute=default_rate,
        downtime_cost_eur_per_minute=per_service,
    )
