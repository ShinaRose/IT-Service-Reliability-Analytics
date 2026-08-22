"""Monthly executive summary: prose written by the LLM, but every number in it must
trace back to metrics computed by relplatform.analytics (never computed by the LLM
itself). See relplatform.ai.numeric_guard for the enforcement mechanism.
"""
from __future__ import annotations

from relplatform.ai.cache import cached_generate_call

EXEC_SUMMARY_SYSTEM = (
    "You are writing a monthly reliability summary for engineering leadership. Use ONLY the "
    "figures given below -- do not calculate, round differently, or introduce any number that "
    "is not explicitly present in the input. Write 3-5 short paragraphs: overall trend, DORA "
    "metrics with their Elite/High/Medium/Low bands, alert noise reduction, and the top risk "
    "areas to prioritize. Cite the specific numbers given for each claim."
)


def build_context(dora_metrics: dict, noise_reduction: dict, risk_df, capacity_forecasts: list[dict], period_label: str) -> str:
    lines = [f"Reporting period: {period_label}", ""]

    lines.append("DORA metrics:")
    for key in ["deployment_frequency", "lead_time_for_changes", "change_failure_rate", "time_to_restore"]:
        m = dora_metrics[key]
        if key == "deployment_frequency":
            val = f"{m['value_per_day']} deploys/day (median {m['value_per_month_median']}/month)"
        elif key == "lead_time_for_changes":
            val = f"median {m['median_hours']} hours"
        elif key == "change_failure_rate":
            val = f"{m['value_pct']}%"
        else:
            val = f"median {m['median_hours']} hours"
        lines.append(f"- {key}: {val}, band={m['band']}, trend={m['trend']['direction']} ({m['trend']['pct_change']}% change)")

    lines.append("")
    lines.append("Alert noise reduction:")
    lines.append(
        f"- {noise_reduction['n_alerts']} raw alerts deduplicated into {noise_reduction['n_distinct_clusters']} "
        f"distinct clusters ({noise_reduction['n_grouped_clusters']} grouped incidents, "
        f"{noise_reduction['n_noise_singletons']} unclustered singletons); "
        f"noise reduction rate = {round(noise_reduction['noise_reduction_rate']*100, 1)}%"
    )

    lines.append("")
    lines.append("Top risk services (risk_score out of 100):")
    for _, row in risk_df.head(5).iterrows():
        lines.append(
            f"- {row['service']}: risk_score={round(row['risk_score'],1)}, "
            f"incidents/month={round(row['incidents_per_month'],2)}, "
            f"MTTR p90={round(row['mttr_p90_minutes'],1)} min, "
            f"change_failure_rate={round(row['change_failure_rate']*100,1)}%"
        )

    lines.append("")
    lines.append("Capacity forecasts:")
    for f in capacity_forecasts:
        if f.get("status") == "breach_projected":
            lines.append(
                f"- {f['service']}: {f['metric']} at {f['current_value']}%, "
                f"projected to breach {f['threshold']}% by {f['projected_breach_date']}"
            )
        else:
            lines.append(f"- {f['service']}: {f['metric']} status={f.get('status')}")

    return "\n".join(lines)


def generate_exec_summary(con, provider, dora_metrics: dict, noise_reduction: dict, risk_df, capacity_forecasts: list[dict], period_label: str):
    context = build_context(dora_metrics, noise_reduction, risk_df, capacity_forecasts, period_label)
    prompt = f"Input figures:\n\n{context}\n\nWrite the summary now."
    text, stats = cached_generate_call(con, provider, task="exec_summary", prompt=prompt, system=EXEC_SUMMARY_SYSTEM, max_tokens=700)
    return text, context, stats
