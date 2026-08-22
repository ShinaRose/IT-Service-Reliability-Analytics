"""Generates varied, genuine-reading multi-paragraph postmortem text per root-cause category.

Each category has multiple phrasing variants for every paragraph slot; the generator
samples one variant per slot and fills in incident-specific details (service, numbers,
durations) so no two postmortems are identical even within the same category. This
variety matters for the AI root-cause-categorization eval: a keyword baseline should
struggle on some of these while the LLM should generalize.
"""
from __future__ import annotations

import random

CATEGORIES = [
    "deployment_regression",
    "resource_exhaustion",
    "third_party_dependency",
    "configuration_error",
    "database_issue",
    "network_partition",
    "traffic_spike",
    "human_error",
]

CONTRIBUTING_FACTORS = {
    "deployment_regression": [
        "insufficient canary analysis before full rollout",
        "no automated rollback trigger on error-rate regression",
        "load test suite did not cover the changed code path",
        "feature flag was enabled for 100% of traffic instead of a gradual ramp",
    ],
    "resource_exhaustion": [
        "connection pool size was not scaled with recent traffic growth",
        "a slow memory leak in a background worker went undetected for weeks",
        "autoscaling policy had a ceiling that was never revisited",
        "garbage collection pauses compounded under sustained load",
    ],
    "third_party_dependency": [
        "no circuit breaker was configured for the upstream vendor call",
        "timeout values for the vendor API were set too high",
        "the vendor's status page did not reflect the outage for 40 minutes",
        "retries without backoff amplified load on the degraded vendor",
    ],
    "configuration_error": [
        "the change was applied directly in the environment instead of through the pipeline",
        "config validation did not catch the malformed value before rollout",
        "two teams pushed conflicting configuration within the same maintenance window",
        "the config change was not peer reviewed",
    ],
    "database_issue": [
        "a missing index caused a full table scan under production data volumes",
        "a long-running migration held a lock longer than anticipated",
        "replica lag exceeded the read-after-write consistency assumption in application code",
        "connection pool exhaustion cascaded from a single slow query",
    ],
    "network_partition": [
        "cross-AZ latency spiked during a cloud provider maintenance event",
        "a misconfigured security group silently dropped a subset of traffic",
        "DNS resolution for an internal service intermittently failed",
        "a BGP route flap affected connectivity to one availability zone",
    ],
    "traffic_spike": [
        "a marketing campaign was not communicated to the on-call team in advance",
        "rate limiting was not tuned for the observed request pattern",
        "cache hit rate dropped sharply once the working set exceeded cache capacity",
        "a retry storm from clients amplified the original spike",
    ],
    "human_error": [
        "a runbook step was executed against the wrong environment",
        "the change was made outside of the standard deployment window without a second approver",
        "a manual database update skipped the usual backup step",
        "an engineer misread a dashboard and restarted the wrong service",
    ],
}

PREVENTIVE_ACTIONS = {
    "deployment_regression": [
        "extend the canary stage to a minimum of 30 minutes with automated error-budget gating",
        "add an automatic rollback trigger keyed to error-rate and latency SLOs",
        "expand load test coverage to include the affected request path",
    ],
    "resource_exhaustion": [
        "right-size the connection pool and add alerting at 70% utilization",
        "add a scheduled memory-leak regression test to the release pipeline",
        "raise the autoscaling ceiling and review it quarterly against traffic growth",
    ],
    "third_party_dependency": [
        "add a circuit breaker with sane timeout and fallback behavior for the vendor call",
        "subscribe to the vendor's status API rather than polling their public status page",
        "cap retries with exponential backoff and jitter",
    ],
    "configuration_error": [
        "require all configuration changes to go through the deployment pipeline with review",
        "add schema validation for configuration values before rollout",
        "add a change calendar so overlapping config changes are visible across teams",
    ],
    "database_issue": [
        "add the missing index and add a query-plan check to the migration review checklist",
        "cap migration lock duration and require online-migration tooling for large tables",
        "add read-after-write consistency checks or route those reads to the primary",
    ],
    "network_partition": [
        "add multi-AZ redundancy for the affected path with automatic failover",
        "add security group change review with a pre-deploy connectivity check",
        "add DNS resolution monitoring with alerting on failure rate",
    ],
    "traffic_spike": [
        "require marketing campaigns above a traffic threshold to be flagged to on-call in advance",
        "retune rate limits based on the observed traffic pattern and add burst headroom",
        "increase cache capacity and add cache hit-rate alerting",
    ],
    "human_error": [
        "add an environment confirmation step to the runbook before destructive actions",
        "require a second approver for changes made outside the standard deployment window",
        "add a pre-flight backup step that cannot be skipped in the runbook tooling",
    ],
}

_SUMMARY_OPENERS = [
    "On {date}, {service} experienced a {sev_desc} that affected {impact}.",
    "Starting at {time} on {date}, customers using {service} began experiencing {impact}.",
    "This postmortem covers a {sev_desc} in {service} that began at {time} on {date} and affected {impact}.",
]

_IMPACT_PHRASES = [
    "checkout and downstream flows",
    "read and write requests for a subset of users",
    "all traffic routed through the affected region",
    "approximately {pct}% of requests during the incident window",
    "requests originating from the mobile client",
]

_CAUSE_TEMPLATES = {
    "deployment_regression": [
        "The root cause was a regression introduced by deploy {deploy_ref}, which changed {area} and "
        "did not account for {edge_case} under production traffic patterns. The new code path was exercised "
        "far more heavily in production than in staging, and error rates began climbing within minutes of rollout.",
        "A deploy to {service} at {deploy_time} introduced a change to {area}. The change passed CI but the "
        "test suite did not reproduce the production traffic shape, and the regression only became visible "
        "once real user traffic hit the new code path.",
    ],
    "resource_exhaustion": [
        "{resource} on {service} climbed steadily over the preceding {lead_hours} hours before crossing a "
        "critical threshold, at which point request latency degraded sharply and a portion of requests began "
        "timing out. Once the primary instances were saturated, retries from upstream callers made the "
        "situation worse.",
        "The proximate cause was exhaustion of {resource} on {service}. Utilization had been trending upward "
        "for {lead_hours} hours, likely due to {trend_cause}, but no alert fired until the pool was already "
        "at capacity.",
    ],
    "third_party_dependency": [
        "{vendor}, a third-party dependency used by {service} for {vendor_purpose}, began returning elevated "
        "error rates and increased latency. Because {service} calls {vendor} synchronously on the critical "
        "path, the degradation propagated directly into user-facing latency and error rates.",
        "An outage at {vendor} degraded {vendor_purpose} for {service}. The dependency is not currently "
        "wrapped in a circuit breaker, so failed calls consumed request threads until the service itself "
        "became unresponsive.",
    ],
    "configuration_error": [
        "A configuration change to {area} on {service} contained {mistake}. The change was applied during "
        "a routine maintenance window and was not caught by existing validation, causing {effect} almost "
        "immediately after rollout.",
        "During a configuration update to {service}, {mistake} was introduced. Because configuration changes "
        "bypass the standard deployment pipeline, no automated check flagged the issue before it took effect.",
    ],
    "database_issue": [
        "A query pattern against the {service} primary database began performing poorly after {trigger}, "
        "leading to {effect}. The database's CPU and connection pool were both driven to saturation, and "
        "the effects were visible to any request that touched the affected table.",
        "A migration executed against the {service} database held a lock for longer than expected because "
        "{trigger}. Writes to the affected table queued up, and the resulting backlog produced {effect} for "
        "downstream requests.",
    ],
    "network_partition": [
        "Connectivity between {service} and its dependencies degraded due to {trigger}. Requests that crossed "
        "the affected network path began timing out, and because retries were not backed off, the retry "
        "traffic itself added load to an already-degraded path.",
        "A network-level fault ({trigger}) caused intermittent packet loss affecting {service}. The "
        "intermittent nature of the fault made it difficult to distinguish from an application-level issue "
        "during the first several minutes of triage.",
    ],
    "traffic_spike": [
        "{service} received a surge of traffic roughly {multiplier}x its typical baseline, driven by "
        "{trigger}. Existing rate limits and autoscaling were not tuned for a spike of this shape, and "
        "request queues began to back up within minutes.",
        "An unanticipated spike in traffic to {service}, attributable to {trigger}, exceeded the provisioned "
        "capacity for the affected tier. Cache hit rate fell as the working set outgrew cache capacity, "
        "pushing additional load onto the origin service.",
    ],
    "human_error": [
        "During a routine operational task on {service}, {mistake}. The action had immediate, visible impact "
        "and was only caught once monitoring alerted on the resulting error rate.",
        "An engineer performing {trigger} on {service} {mistake}. Because the action was performed outside "
        "the standard change process, it was not peer reviewed before being carried out.",
    ],
}

_DETECTION_TEMPLATES = [
    "The incident was detected by automated alerting {detect_delay} minutes after onset, when {alert_trigger} "
    "crossed its configured threshold. On-call acknowledged within {ack_minutes} minutes and began triage "
    "immediately, correlating the alert storm across {n_services} affected services before identifying {service} "
    "as the origin.",
    "Automated monitoring fired {detect_delay} minutes into the incident on {alert_trigger}. The on-call "
    "engineer acknowledged in {ack_minutes} minutes; initial triage was complicated by a wave of correlated "
    "alerts from downstream services before the team traced the issue back to {service}.",
]

_RESOLUTION_TEMPLATES = [
    "The team mitigated impact by {mitigation}, which brought error rates back to baseline within "
    "{restore_minutes} minutes of the mitigation being applied. Full resolution, including validation that "
    "downstream services had recovered, was confirmed {total_minutes} minutes after the incident began.",
    "Mitigation was applied by {mitigation}. Metrics returned to baseline {restore_minutes} minutes later, "
    "and the incident was formally closed {total_minutes} minutes after it started once all downstream "
    "alerts had cleared.",
]

_MITIGATIONS = {
    "deployment_regression": ["rolling back deploy {deploy_ref}", "reverting the offending change and redeploying the previous version"],
    "resource_exhaustion": ["restarting the affected instances and temporarily raising the resource ceiling", "manually scaling out the affected service"],
    "third_party_dependency": ["failing over to the backup provider", "enabling a temporary in-house fallback path"],
    "configuration_error": ["reverting the configuration change", "rolling forward with the corrected configuration"],
    "database_issue": ["killing the offending query and adding an emergency index", "aborting the migration and restoring normal write traffic"],
    "network_partition": ["rerouting traffic away from the affected availability zone", "failing over to the secondary network path"],
    "traffic_spike": ["enabling emergency rate limiting", "scaling out the affected tier and warming the cache"],
    "human_error": ["reverting the erroneous action", "restoring from the most recent known-good state"],
}

_VENDORS = ["StripeConnectAPI", "GeoIP-Cloud", "SendGridRelay", "TwilioVoice", "CloudCDN-Edge", "TaxJarCalc"]
_VENDOR_PURPOSE = {
    "StripeConnectAPI": "payment authorization",
    "GeoIP-Cloud": "geolocation lookups",
    "SendGridRelay": "transactional email delivery",
    "TwilioVoice": "two-factor SMS delivery",
    "CloudCDN-Edge": "static asset delivery",
    "TaxJarCalc": "tax calculation",
}


def _sev_desc(severity: str) -> str:
    return {
        "SEV1": "critical, customer-facing outage",
        "SEV2": "major service degradation",
        "SEV3": "minor service degradation",
        "SEV4": "low-impact anomaly",
    }[severity]


def generate_postmortem(
    rng: random.Random,
    service: str,
    severity: str,
    category: str,
    started_at,
    resolved_at,
    ack_minutes: int,
    n_alert_services: int,
    deploy_ref: str | None,
) -> tuple[str, list[str], str]:
    """Returns (postmortem_text, contributing_factors[2], preventive_action)."""
    total_minutes = max(1, int((resolved_at - started_at).total_seconds() / 60))
    restore_minutes = max(1, int(total_minutes * rng.uniform(0.5, 0.9)))
    detect_delay = max(1, int(rng.uniform(1, 8)))

    vendor = rng.choice(_VENDORS)
    ctx = {
        "date": started_at.strftime("%B %d, %Y"),
        "time": started_at.strftime("%H:%M UTC"),
        "service": service,
        "sev_desc": _sev_desc(severity),
        "impact": rng.choice(_IMPACT_PHRASES).format(pct=rng.randint(8, 65)),
        "deploy_ref": deploy_ref or f"dep-{rng.randint(1000,9999)}",
        "deploy_time": started_at.strftime("%H:%M UTC"),
        "area": rng.choice(["the request-validation middleware", "the caching layer", "the retry logic",
                             "the checkout pricing calculation", "the session-handling code", "the rate limiter"]),
        "edge_case": rng.choice(["null values in an optional field", "a burst of concurrent requests",
                                  "requests missing a legacy header", "a large-payload request body"]),
        "resource": rng.choice(["the database connection pool", "heap memory", "the thread pool",
                                 "file descriptors", "the message queue"]),
        "lead_hours": rng.randint(2, 36),
        "trend_cause": rng.choice(["a slow leak in a recently deployed background job",
                                    "organic traffic growth that outpaced capacity planning",
                                    "a scheduled batch job that was not accounted for in capacity planning"]),
        "vendor": vendor,
        "vendor_purpose": _VENDOR_PURPOSE[vendor],
        "mistake": rng.choice(["a typo in a numeric threshold", "an inverted boolean flag",
                                "a stale credential that had already been rotated",
                                "a value copied from the wrong environment"]),
        "effect": rng.choice(["a sharp rise in p99 latency", "elevated 5xx error rates",
                               "request timeouts for a subset of users", "a queue backlog that grew unbounded"]),
        "trigger": rng.choice(["an unannounced schema migration", "a spike in concurrent long-running reports",
                                "a cloud provider maintenance event", "a misconfigured load balancer health check",
                                "a promotional email sent to the full customer list",
                                "a partner integration retrying aggressively after its own outage",
                                "a decommissioned database replica whose DNS entry was not yet removed",
                                "an emergency data backfill run without a maintenance window"]),
        "multiplier": round(rng.uniform(3, 12), 1),
        "detect_delay": detect_delay,
        "alert_trigger": rng.choice(["error rate", "p99 latency", "5xx rate", "queue depth", "CPU utilization"]),
        "ack_minutes": ack_minutes,
        "n_services": n_alert_services,
        "restore_minutes": restore_minutes,
        "total_minutes": total_minutes,
    }

    summary = rng.choice(_SUMMARY_OPENERS).format(**ctx)
    cause = rng.choice(_CAUSE_TEMPLATES[category]).format(**ctx)
    detection = rng.choice(_DETECTION_TEMPLATES).format(**ctx)
    mitigation_text = rng.choice(_MITIGATIONS[category]).format(**ctx)
    ctx["mitigation"] = mitigation_text
    resolution = rng.choice(_RESOLUTION_TEMPLATES).format(**ctx)

    factors = rng.sample(CONTRIBUTING_FACTORS[category], k=2)
    action = rng.choice(PREVENTIVE_ACTIONS[category])

    paragraphs = [
        f"## Summary\n{summary}",
        f"## Root Cause\n{cause}",
        f"## Detection & Response\n{detection}",
        f"## Resolution\n{resolution}",
        "## Contributing Factors\n" + "\n".join(f"- {f}" for f in factors),
        f"## Preventive Action\n{action.capitalize()}.",
    ]
    return "\n\n".join(paragraphs), factors, action
