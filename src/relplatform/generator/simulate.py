"""End-to-end synthetic data simulation: deployments, incidents, alert storms, resource metrics.

Ground truth baked in on purpose so downstream analytics have something real to recover:
- deployments carry a latent risk (size, timing, service baseline) that drives whether an
  incident follows -- this is what the change-failure logistic regression should learn.
- each service has distinct MTTR distribution parameters -- this is what the per-service
  MTTR fit should recover.
- alerts carry a ground-truth `incident_id` (eval-only, never fed to the clustering step)
  plus unrelated background noise alerts, so clustering has a real noise-reduction signal
  and a real ARI/purity to be scored against.
- two services get an upward-trending resource metric that crosses a breach threshold
  within the forecast horizon; the rest stay flat, so capacity forecasting has a real
  answer instead of always "someday".
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from relplatform.generator.alert_messages import sample_alert
from relplatform.generator.graph import CALL_EDGES, SERVICE_META, build_graph, callers_by_hop
from relplatform.generator.postmortems import CATEGORIES, generate_postmortem

SEVERITIES = ["SEV1", "SEV2", "SEV3", "SEV4"]

MTTR_LOGNORMAL_PARAMS = {  # (mu, sigma) of log(minutes-to-resolve)
    "api-gateway": (3.8, 0.55),
    "auth-service": (3.55, 0.50),
    "checkout-service": (4.25, 0.65),
    "payments-service": (4.60, 0.75),
    "inventory-service": (3.95, 0.55),
    "notification-service": (3.25, 0.45),
    "search-service": (4.05, 0.60),
    "recommendation-service": (4.35, 0.60),
}

NON_DEPLOY_CATEGORIES = [c for c in CATEGORIES if c not in ("deployment_regression", "configuration_error")]
NON_DEPLOY_WEIGHTS = [0.22, 0.16, 0.18, 0.12, 0.14, 0.18]  # resource_exhaustion, third_party, database, network, traffic, human_error


@dataclass
class SimResult:
    deployments: list[dict] = field(default_factory=list)
    incidents: list[dict] = field(default_factory=list)
    alerts: list[dict] = field(default_factory=list)
    resource_metrics: list[dict] = field(default_factory=list)


def _sample_deploy_times(rng: random.Random, start: datetime, end: datetime, per_week: float) -> list[datetime]:
    days = (end - start).days
    n = rng.gauss(per_week * days / 7, max(1.0, per_week * days / 7 * 0.1))
    n = max(0, int(round(n)))
    times = []
    for _ in range(n):
        offset_days = rng.uniform(0, days)
        t = start + timedelta(days=offset_days)
        # bias toward weekday business hours (09:00-18:00), with a tail of off-hours deploys
        if rng.random() < 0.8:
            hour = rng.triangular(9, 18, 13)
        else:
            hour = rng.uniform(0, 24)
        t = t.replace(hour=int(hour), minute=rng.randint(0, 59), second=rng.randint(0, 59))
        times.append(t)
    return sorted(times)


def _deploy_risk(lines_changed: int, commit_count: int, is_weekend: bool, is_off_hours: bool, baseline_risk: float) -> float:
    # Calibrated (against the actual lines_changed/commit_count distributions the
    # generator produces) so a median deploy lands ~5%, a large off-hours weekend
    # deploy on a risky service reaches ~55%, and a small safe deploy is ~2% -- wide
    # enough spread that the probability itself is a real, separable signal (checked
    # via the Bayes-optimal ROC AUC of p_incident against the sampled outcome, not just
    # eyeballed), which is what gives the change-failure model something to learn.
    z = (
        -4.95
        + 0.213 * np.log1p(lines_changed)
        + 0.05 * commit_count
        + 1.30 * is_weekend
        + 0.95 * is_off_hours
        + 6.0 * baseline_risk
    )
    return 1.0 / (1.0 + np.exp(-z))


def simulate(seed: int, months: int = 12) -> SimResult:
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    start = end - timedelta(days=30 * months)

    g = build_graph()
    result = SimResult()

    deploy_counter = 0
    incident_counter = 0
    alert_counter = 0

    all_incident_specs = []  # (started_candidate, service, category, triggering_deploy_id, origin_service, severity_hint)

    # ---- deployments + deploy-triggered incidents ----
    for service, meta in SERVICE_META.items():
        times = _sample_deploy_times(rng, start, end, meta["deploys_per_week"])
        for t in times:
            deploy_counter += 1
            dep_id = f"DEP-{deploy_counter:05d}"
            commit_count = max(1, int(nprng.poisson(3) + 1))
            lines_changed = max(3, int(nprng.lognormal(mean=4.2, sigma=1.1) * (0.6 + 0.2 * commit_count)))
            lead_time_hours = max(0.05, float(nprng.lognormal(
                mean=(0.3 if meta["deploys_per_week"] > 10 else 2.2 if meta["deploys_per_week"] > 2 else 3.6),
                sigma=0.7,
            )))
            is_weekend = t.weekday() >= 5
            is_off_hours = t.hour < 7 or t.hour >= 20

            p_incident = _deploy_risk(lines_changed, commit_count, is_weekend, is_off_hours, meta["baseline_risk"])
            triggers_incident = rng.random() < p_incident
            rolled_back = False

            if triggers_incident:
                incident_counter += 1
                inc_id = f"INC-{incident_counter:05d}"
                category = "deployment_regression" if rng.random() < 0.78 else "configuration_error"
                delay = timedelta(minutes=rng.uniform(4, 150))
                started_at = t + delay
                if started_at < end:
                    severity_hint = rng.betavariate(2, 3)  # decoupled from trigger probability: how bad it is, given it happened
                    rolled_back = rng.random() < (0.75 if severity_hint > 0.5 else 0.35)
                    all_incident_specs.append(dict(
                        inc_id=inc_id, started_at=started_at, service=service, category=category,
                        triggering_deploy_id=dep_id, origin_service=service, severity_hint=severity_hint,
                    ))

            result.deployments.append(dict(
                id=dep_id, service=service, deployed_at=t, commit_count=commit_count,
                lines_changed=lines_changed, lead_time_hours=round(lead_time_hours, 2),
                is_weekend=is_weekend, is_off_hours=is_off_hours, rolled_back=rolled_back,
            ))

    # ---- independent (non-deploy-triggered) incidents ----
    for service, meta in SERVICE_META.items():
        incidents_per_month = 1.3 + meta["baseline_risk"] * 3.5
        n = max(0, int(round(nprng.poisson(incidents_per_month * months))))
        for _ in range(n):
            incident_counter += 1
            inc_id = f"INC-{incident_counter:05d}"
            offset_days = rng.uniform(0, (end - start).days)
            started_at = start + timedelta(days=offset_days, hours=rng.uniform(0, 24))
            category = rng.choices(NON_DEPLOY_CATEGORIES, weights=NON_DEPLOY_WEIGHTS, k=1)[0]
            severity_hint = rng.betavariate(2, 4)
            all_incident_specs.append(dict(
                inc_id=inc_id, started_at=started_at, service=service, category=category,
                triggering_deploy_id=None, origin_service=service, severity_hint=severity_hint,
            ))

    all_incident_specs.sort(key=lambda x: x["started_at"])

    # ---- flesh out incidents + alert storms ----
    for spec in all_incident_specs:
        service = spec["service"]
        h = spec["severity_hint"]
        w1, w2, w3, w4 = h ** 2, h, (1 - h), (1 - h) ** 1.5 * 0.7
        severity = rng.choices(SEVERITIES, weights=[w1, w2, w3, w4], k=1)[0]

        mu, sigma = MTTR_LOGNORMAL_PARAMS[service]
        sev_mult = {"SEV1": 1.35, "SEV2": 1.1, "SEV3": 0.85, "SEV4": 0.6}[severity]
        mttr_minutes = max(3.0, float(nprng.lognormal(mean=mu, sigma=sigma)) * sev_mult)
        started_at = spec["started_at"]
        resolved_at = started_at + timedelta(minutes=mttr_minutes)

        ack_mu = {"SEV1": 0.9, "SEV2": 1.4, "SEV3": 2.0, "SEV4": 2.4}[severity]
        ack_minutes = max(1.0, float(nprng.lognormal(mean=ack_mu, sigma=0.5)))
        ack_minutes = min(ack_minutes, mttr_minutes * 0.8)
        acknowledged_at = started_at + timedelta(minutes=ack_minutes)

        hops = callers_by_hop(g, service, max_hops=2)
        n_alert_services = 1 + sum(len(v) for v in hops.values())

        postmortem_text, factors, action = generate_postmortem(
            rng, service, severity, spec["category"], started_at, resolved_at,
            int(ack_minutes), n_alert_services, spec["triggering_deploy_id"],
        )

        result.incidents.append(dict(
            id=spec["inc_id"], service=service, severity=severity, started_at=started_at,
            acknowledged_at=acknowledged_at, resolved_at=resolved_at,
            root_cause_category=spec["category"], postmortem_text=postmortem_text,
            triggering_deploy_id=spec["triggering_deploy_id"], origin_service=spec["origin_service"],
        ))

        # --- alert storm ---
        base_n = rng.randint(40, 200)
        sev_scale = {"SEV1": 1.0, "SEV2": 0.85, "SEV3": 0.6, "SEV4": 0.4}[severity]
        n_alerts = max(40, int(base_n * sev_scale))

        service_weights = [(service, 1.0)]
        for hop, svcs in hops.items():
            w = 0.45 if hop == 1 else 0.18
            service_weights += [(s, w) for s in svcs]
        svcs, weights = zip(*service_weights)

        window = max(1.0, (resolved_at - started_at).total_seconds())
        for _ in range(n_alerts):
            alert_counter += 1
            a_id = f"ALT-{alert_counter:06d}"
            a_service = rng.choices(svcs, weights=weights, k=1)[0]
            # front-loaded arrival: exponential-ish skew toward the start of the incident
            frac = min(0.999, nprng.beta(1.5, 4.0))
            fired_at = started_at + timedelta(seconds=frac * window)
            storm_intensity = max(0.05, 1.0 - frac) * sev_scale
            message, alert_sev = sample_alert(rng, a_service, storm_intensity, category=spec["category"])
            duration = timedelta(minutes=rng.uniform(1, 20))
            a_resolved = min(resolved_at, fired_at + duration)
            result.alerts.append(dict(
                id=a_id, service=a_service, severity=alert_sev, message=message,
                fired_at=fired_at, resolved_at=a_resolved, incident_id=spec["inc_id"],
            ))

    # ---- background noise alerts (unrelated to any incident) ----
    n_noise = int(len(result.alerts) * 0.07)
    services = list(SERVICE_META.keys())
    for _ in range(n_noise):
        alert_counter += 1
        a_id = f"ALT-{alert_counter:06d}"
        a_service = rng.choice(services)
        offset_days = rng.uniform(0, (end - start).days)
        fired_at = start + timedelta(days=offset_days, hours=rng.uniform(0, 24))
        message, alert_sev = sample_alert(rng, a_service, storm_intensity=rng.uniform(0.0, 0.25))
        a_resolved = fired_at + timedelta(minutes=rng.uniform(1, 15))
        result.alerts.append(dict(
            id=a_id, service=a_service, severity=alert_sev, message=message,
            fired_at=fired_at, resolved_at=a_resolved, incident_id=None,
        ))

    result.alerts.sort(key=lambda a: a["fired_at"])

    # ---- resource metrics (capacity forecasting substrate) ----
    trending_services = {"payments-service": 0.11, "checkout-service": 0.07}
    days = (end - start).days
    for service in SERVICE_META:
        slope = trending_services.get(service, rng.uniform(-0.01, 0.015))
        base = rng.uniform(35, 50)
        for d in range(days):
            ts = start + timedelta(days=d)
            weekly = 4 * np.sin(2 * np.pi * d / 7)
            noise = nprng.normal(0, 2.5)
            value = base + slope * d + weekly + noise
            value = float(np.clip(value, 1, 100))
            result.resource_metrics.append(dict(
                service=service, ts=ts, metric_name="db_connection_pool_utilization_pct", value=round(value, 2),
            ))

    # p95 latency: a genuine (if synthetic) latency signal, not just a declared-but-
    # unmeasured config target -- SLO latency budgets in relplatform.slo need something
    # to check against. Baseline per service + weekly seasonality + noise, with a
    # severity-scaled spike on any day that service had an active incident, so the SLO
    # burn-rate math has real correlated signal to detect rather than pure noise.
    _SEV_LATENCY_MULT = {"SEV1": 3.0, "SEV2": 2.0, "SEV3": 1.4, "SEV4": 1.15}
    incidents_by_service: dict[str, list[tuple[datetime, datetime, str]]] = {}
    for inc in result.incidents:
        incidents_by_service.setdefault(inc["service"], []).append(
            (inc["started_at"], inc["resolved_at"], inc["severity"])
        )

    for service in SERVICE_META:
        baseline_ms = rng.uniform(80, 320)
        svc_incidents = incidents_by_service.get(service, [])
        for d in range(days):
            day_start = start + timedelta(days=d)
            day_end = day_start + timedelta(days=1)
            weekly = 0.05 * baseline_ms * np.sin(2 * np.pi * d / 7)
            noise = nprng.normal(0, baseline_ms * 0.06)

            spike_mult = 1.0
            for inc_start, inc_end, sev in svc_incidents:
                if inc_start < day_end and inc_end > day_start:
                    spike_mult = max(spike_mult, _SEV_LATENCY_MULT.get(sev, 1.0))

            value = max(5.0, (baseline_ms + weekly + noise) * spike_mult)
            result.resource_metrics.append(dict(
                service=service, ts=day_start, metric_name="p95_latency_ms", value=round(float(value), 1),
            ))

    return result
