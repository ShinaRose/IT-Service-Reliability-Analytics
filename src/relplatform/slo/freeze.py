"""Ship/freeze traffic light per service, cross-checked against the existing
change-failure model rather than treated as the only signal.

Traffic light (SLO-only):
  green  -- budget consumed < 50% and no burn-rate alert firing
  yellow -- budget consumed 50-100%, or the slow-burn alert is firing
  red    -- budget exhausted, or the fast-burn alert is firing

The cross-check: relplatform.analytics.change_failure already estimates, per service,
the probability a deploy causes an incident (from deploy_risk_scores / the labeled
deploy history). A service can be SLO-red for reasons that have nothing to do with
deploy risk (a capacity issue, a third-party dependency, a single big incident already
priced into the window) -- freezing deploys wouldn't fix that. Conversely a service can
be SLO-green while its recent deploys are trending risky, and the budget just hasn't
caught up yet. Both cases are worth surfacing explicitly rather than collapsing into one
number.
"""
from __future__ import annotations

from dataclasses import dataclass

from relplatform.slo.budget import ErrorBudgetStatus
from relplatform.slo.burn_rate import BurnRateAlert

# A service's historical deploy-caused-incident rate above this is "the change-failure
# model considers this service risky". 15% is the Elite/High DORA change-failure-rate
# boundary already used elsewhere in this codebase (relplatform.config.DORA_BANDS) --
# reused here so the two signals are being compared on a consistent bar, not two
# arbitrarily different thresholds.
CHANGE_FAILURE_RISK_THRESHOLD_PCT = 15.0


@dataclass
class FreezeRecommendation:
    service: str
    light: str  # "green" | "yellow" | "red"
    reasons: list[str]
    change_failure_rate_pct: float | None
    change_failure_model_flags_risky: bool | None
    disagreement: bool
    disagreement_note: str | None


def recommend(
    budget: ErrorBudgetStatus, burn_alerts: list[BurnRateAlert], change_failure_rate_pct: float | None = None,
) -> FreezeRecommendation:
    fast = next((a for a in burn_alerts if a.rule_name == "fast_burn"), None)
    slow = next((a for a in burn_alerts if a.rule_name == "slow_burn"), None)

    reasons = []
    consumed = budget.budget_consumed_pct or 0.0

    if budget.exhausted:
        light = "red"
        reasons.append(f"error budget exhausted ({consumed:.0f}% consumed)")
    elif fast is not None and fast.firing:
        light = "red"
        reasons.append(f"fast-burn alert firing (burn rate {fast.long_window_burn_rate} >= threshold {fast.threshold})")
    elif consumed >= 50 or (slow is not None and slow.firing):
        light = "yellow"
        if consumed >= 50:
            reasons.append(f"{consumed:.0f}% of error budget consumed")
        if slow is not None and slow.firing:
            reasons.append(f"slow-burn alert firing (burn rate {slow.long_window_burn_rate} >= threshold {slow.threshold})")
    else:
        light = "green"
        reasons.append(f"only {consumed:.0f}% of error budget consumed, no burn-rate alerts firing")

    model_flags_risky = None
    disagreement = False
    disagreement_note = None
    if change_failure_rate_pct is not None:
        model_flags_risky = change_failure_rate_pct >= CHANGE_FAILURE_RISK_THRESHOLD_PCT
        slo_says_freeze = light == "red"
        if slo_says_freeze and not model_flags_risky:
            disagreement = True
            disagreement_note = (
                f"SLO says freeze, but the change-failure model rates deploys to this service as "
                f"low-risk ({change_failure_rate_pct:.1f}% historical deploy-caused-incident rate). "
                f"The SLO burn may be driven by something other than deploy risk (capacity, a "
                f"third-party dependency, a single large incident) -- freezing deploys may not "
                f"address the actual cause."
            )
        elif not slo_says_freeze and model_flags_risky:
            disagreement = True
            disagreement_note = (
                f"SLO says {light}, but the change-failure model rates deploys to this service as "
                f"risky ({change_failure_rate_pct:.1f}% historical deploy-caused-incident rate). "
                f"The error budget hasn't caught up to that risk yet -- worth caution even though "
                f"the budget itself looks fine right now."
            )

    return FreezeRecommendation(
        service=budget.service, light=light, reasons=reasons,
        change_failure_rate_pct=change_failure_rate_pct, change_failure_model_flags_risky=model_flags_risky,
        disagreement=disagreement, disagreement_note=disagreement_note,
    )
