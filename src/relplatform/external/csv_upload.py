"""Bring-your-own-data CSV upload: column-mapping and schema validation that turns
arbitrary user CSVs into the same deployments/incidents DataFrame shape the synthetic
pipeline and the GitHub connector both already produce, then runs them through the same
relplatform.analytics.dora functions -- one DORA scoring path for all three data
sources (synthetic, GitHub, uploaded), not three separate ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from relplatform.analytics.dora import change_failure_rate as _change_failure_rate_bands
from relplatform.analytics.dora import deployment_frequency as _deployment_frequency_bands
from relplatform.analytics.dora import label_deploy_caused_incidents
from relplatform.analytics.dora import lead_time_for_changes as _lead_time_bands
from relplatform.analytics.dora import time_to_restore as _time_to_restore_bands

DEFAULT_SERVICE = "uploaded"


@dataclass
class ValidationError:
    field: str
    message: str


@dataclass
class MappedDeployments:
    df: pd.DataFrame
    errors: list[ValidationError] = field(default_factory=list)


@dataclass
class MappedIncidents:
    df: pd.DataFrame
    errors: list[ValidationError] = field(default_factory=list)


def map_deployments(
    raw: pd.DataFrame, deployed_at_col: str, service_col: str | None = None,
    lead_time_hours_col: str | None = None, commit_at_col: str | None = None,
) -> MappedDeployments:
    """`lead_time_hours_col` and `commit_at_col` are two mutually exclusive ways to get a
    lead time: supply the hours directly, or supply a commit timestamp and let this
    derive deployed_at - commit_at. If neither is mapped, lead time is left unavailable
    rather than fabricated."""
    errors: list[ValidationError] = []
    if deployed_at_col not in raw.columns:
        errors.append(ValidationError("deployed_at", f"Column '{deployed_at_col}' not found in the uploaded file."))
        return MappedDeployments(pd.DataFrame(columns=["deployed_at", "service", "lead_time_hours"]), errors)

    out = pd.DataFrame()
    out["deployed_at"] = pd.to_datetime(raw[deployed_at_col], errors="coerce")
    out["service"] = raw[service_col].astype(str) if service_col else DEFAULT_SERVICE

    if lead_time_hours_col:
        out["lead_time_hours"] = pd.to_numeric(raw[lead_time_hours_col], errors="coerce")
    elif commit_at_col:
        commit_at = pd.to_datetime(raw[commit_at_col], errors="coerce")
        out["lead_time_hours"] = (out["deployed_at"] - commit_at).dt.total_seconds() / 3600
    else:
        out["lead_time_hours"] = pd.NA
        errors.append(ValidationError("lead_time_hours", "No lead-time or commit-time column mapped -- lead time for changes will be unavailable."))

    n_bad_dates = int(out["deployed_at"].isna().sum())
    if n_bad_dates:
        errors.append(ValidationError("deployed_at", f"{n_bad_dates} row(s) could not be parsed as a date and were dropped."))
    out = out.dropna(subset=["deployed_at"]).reset_index(drop=True)

    negative = int((out["lead_time_hours"] < 0).sum())
    if negative:
        errors.append(ValidationError(
            "lead_time_hours",
            f"{negative} row(s) had a negative lead time (deployed before the mapped commit time) -- set to unavailable for those rows, not clipped to zero.",
        ))
        out.loc[out["lead_time_hours"] < 0, "lead_time_hours"] = pd.NA

    return MappedDeployments(out, errors)


def map_incidents(raw: pd.DataFrame, started_at_col: str, resolved_at_col: str, service_col: str | None = None) -> MappedIncidents:
    errors: list[ValidationError] = []
    missing = [c for c in (started_at_col, resolved_at_col) if c not in raw.columns]
    if missing:
        for m in missing:
            errors.append(ValidationError(m, f"Column '{m}' not found in the uploaded file."))
        return MappedIncidents(pd.DataFrame(columns=["started_at", "resolved_at", "service"]), errors)

    out = pd.DataFrame()
    out["started_at"] = pd.to_datetime(raw[started_at_col], errors="coerce")
    out["resolved_at"] = pd.to_datetime(raw[resolved_at_col], errors="coerce")
    out["service"] = raw[service_col].astype(str) if service_col else DEFAULT_SERVICE

    n_bad = int(out[["started_at", "resolved_at"]].isna().any(axis=1).sum())
    if n_bad:
        errors.append(ValidationError("started_at/resolved_at", f"{n_bad} row(s) had an unparseable date and were dropped."))
    out = out.dropna(subset=["started_at", "resolved_at"]).reset_index(drop=True)

    backwards = int((out["resolved_at"] < out["started_at"]).sum())
    if backwards:
        errors.append(ValidationError("resolved_at", f"{backwards} row(s) had resolved_at before started_at and were dropped."))
        out = out[out["resolved_at"] >= out["started_at"]].reset_index(drop=True)

    return MappedIncidents(out, errors)


@dataclass
class UploadedDoraResult:
    deployment_frequency: dict | None
    lead_time_for_changes: dict | None
    change_failure_rate: dict | None
    time_to_restore: dict | None
    n_deployments: int
    n_incidents: int
    errors: list[ValidationError] = field(default_factory=list)


def compute_uploaded_dora(
    deployments: MappedDeployments, incidents: MappedIncidents, deploy_incident_window_hours: float = 4.0,
) -> UploadedDoraResult:
    """Reuses relplatform.analytics.dora's own change-failure heuristic
    (label_deploy_caused_incidents: nearest prior deploy within a time window) rather
    than a separate uploaded-data-only rule -- the same time-proximity judgment call the
    synthetic pipeline already documents and the rest of the app already presents."""
    errors = list(deployments.errors) + list(incidents.errors)
    dep_df, inc_df = deployments.df, incidents.df

    deploy_freq = lead_time = change_failure = time_to_restore = None

    if len(dep_df):
        deploy_freq = _deployment_frequency_bands(dep_df)
        lt_input = dep_df.dropna(subset=["lead_time_hours"])
        if len(lt_input):
            lead_time = _lead_time_bands(lt_input)

        labeled = (
            label_deploy_caused_incidents(dep_df, inc_df, deploy_incident_window_hours)
            if len(inc_df) else dep_df.assign(caused_incident=0)
        )
        change_failure = _change_failure_rate_bands(labeled)

    if len(inc_df):
        restore_hours = (inc_df["resolved_at"] - inc_df["started_at"]).dt.total_seconds() / 3600
        if (restore_hours > 0).any():
            time_to_restore = _time_to_restore_bands(inc_df)

    return UploadedDoraResult(
        deployment_frequency=deploy_freq, lead_time_for_changes=lead_time,
        change_failure_rate=change_failure, time_to_restore=time_to_restore,
        n_deployments=len(dep_df), n_incidents=len(inc_df), errors=errors,
    )
