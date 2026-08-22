"""Varied alert message wording, grouped by underlying signal type.

Real alert storms don't repeat the same string 150 times -- different monitors,
dashboards and thresholds phrase the same underlying problem differently. Each
signal type below has several templates so a single incident's alert storm is
lexically diverse, which is exactly what the dedup/clustering step has to see
past.
"""
from __future__ import annotations

import random

# signal -> list of message templates. {service} and {val} are filled in.
TEMPLATES: dict[str, list[str]] = {
    "error_rate": [
        "[{service}] error rate {val}% over last 5m (threshold 2%)",
        "High 5xx rate detected on {service}: {val}% of requests failing",
        "{service} error budget burn rate critical -- {val}% errors in window",
        "PagerDuty: {service} HTTP 500 rate {val}% exceeds SLO",
        "Elevated failure rate on {service} endpoints ({val}%)",
    ],
    "latency": [
        "[{service}] p99 latency {val}ms, above 500ms SLO",
        "{service} response time degraded: p95 = {val}ms",
        "Latency spike on {service}: requests taking {val}ms on average",
        "SLO breach: {service} p99 latency at {val}ms",
        "{service} slow responses detected, avg {val}ms over 5m",
    ],
    "timeout": [
        "{service} request timeouts increasing, {val} in last 5m",
        "Timeout errors spiking on {service} ({val} events)",
        "[{service}] client timeout rate elevated: {val} occurrences",
        "Upstream timeout from {service}, count={val}",
    ],
    "cpu": [
        "{service} CPU utilization at {val}%",
        "High CPU alert: {service} instances averaging {val}% CPU",
        "[{service}] CPU saturation warning ({val}%)",
    ],
    "memory": [
        "{service} memory usage at {val}%, approaching limit",
        "Memory pressure detected on {service}: {val}% used",
        "[{service}] heap utilization {val}%, GC pauses increasing",
    ],
    "connection_pool": [
        "{service} DB connection pool at {val}% capacity",
        "Connection pool exhaustion warning on {service} ({val}% used)",
        "[{service}] unable to acquire DB connection, pool at {val}%",
    ],
    "queue_depth": [
        "{service} queue depth {val}, consumers falling behind",
        "Backlog growing on {service} queue: {val} messages pending",
        "[{service}] message queue depth alert: {val}",
    ],
    "dependency": [
        "{service} reporting elevated errors calling downstream dependency",
        "[{service}] circuit breaker OPEN for downstream call",
        "{service} health check failing dependency probe",
        "Downstream call failures detected from {service}",
    ],
    "availability": [
        "{service} health check failing on {val}% of instances",
        "[{service}] readiness probe failures, {val}% of pods unhealthy",
        "Service availability degraded for {service}: {val}% instances down",
    ],
}

SEVERITIES = ["critical", "warning", "info"]

# Which alert signal types a given root cause tends to produce -- this is what gives the
# RAG retrieval task (and a human on-call engineer) something real to pattern-match on:
# a resource_exhaustion incident's storm really is dominated by cpu/memory/connection_pool
# alerts, not a uniformly random mix of every signal type. Signals not listed still occur
# (via the small uniform weight below) since real storms are never perfectly clean.
CATEGORY_SIGNAL_WEIGHTS: dict[str, dict[str, float]] = {
    "deployment_regression": {"error_rate": 3, "latency": 3, "timeout": 2},
    "resource_exhaustion": {"cpu": 3, "memory": 3, "connection_pool": 3, "queue_depth": 2},
    "third_party_dependency": {"dependency": 4, "timeout": 3, "latency": 2},
    "configuration_error": {"error_rate": 3, "availability": 3, "dependency": 2},
    "database_issue": {"latency": 3, "connection_pool": 3, "timeout": 2, "queue_depth": 2},
    "network_partition": {"availability": 3, "timeout": 3, "dependency": 2},
    "traffic_spike": {"latency": 3, "queue_depth": 3, "error_rate": 2, "cpu": 2},
    "human_error": {"error_rate": 3, "availability": 2},
}
_BACKGROUND_WEIGHT = 0.6  # small uniform floor over every signal so storms aren't perfectly clean


def _pick_signal(rng: random.Random, category: str | None) -> str:
    signals = list(TEMPLATES.keys())
    if category is None or category not in CATEGORY_SIGNAL_WEIGHTS:
        return rng.choice(signals)
    preferred = CATEGORY_SIGNAL_WEIGHTS[category]
    weights = [preferred.get(s, 0.0) + _BACKGROUND_WEIGHT for s in signals]
    return rng.choices(signals, weights=weights, k=1)[0]


def sample_alert(rng: random.Random, service: str, storm_intensity: float, category: str | None = None) -> tuple[str, str]:
    """Return (message, severity). storm_intensity in [0,1] biases toward higher values/severity.
    `category` (the incident's root cause, when known -- i.e. not for background noise alerts)
    biases which signal type is more likely to fire; see CATEGORY_SIGNAL_WEIGHTS."""
    signal = _pick_signal(rng, category)
    template = rng.choice(TEMPLATES[signal])
    if signal in ("error_rate",):
        val = round(rng.uniform(3, 15) + storm_intensity * rng.uniform(10, 60), 1)
    elif signal == "latency":
        val = int(rng.uniform(500, 900) + storm_intensity * rng.uniform(200, 4000))
    elif signal in ("cpu", "memory", "connection_pool"):
        val = int(min(100, rng.uniform(70, 85) + storm_intensity * rng.uniform(5, 20)))
    elif signal == "queue_depth":
        val = int(rng.uniform(50, 500) + storm_intensity * rng.uniform(200, 5000))
    elif signal == "timeout":
        val = int(rng.uniform(3, 20) + storm_intensity * rng.uniform(10, 150))
    elif signal == "availability":
        val = int(min(100, rng.uniform(5, 30) + storm_intensity * rng.uniform(10, 60)))
    else:
        val = None

    message = template.format(service=service, val=val) if "{val}" in template else template.format(service=service)

    sev_roll = rng.random()
    if storm_intensity > 0.6 and sev_roll < 0.55:
        severity = "critical"
    elif sev_roll < 0.3:
        severity = "critical"
    elif sev_roll < 0.75:
        severity = "warning"
    else:
        severity = "info"
    return message, severity
