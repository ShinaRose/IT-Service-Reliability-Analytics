"""Root-cause categorization eval: hand-labeled postmortems vs. a keyword baseline vs.
the AI task.

"Hand-labeled" here means the generator's own ground-truth `root_cause_category` for a
stratified sample of incidents -- for synthetic data this plays the same role a human
labeler would: it's the answer key the postmortem text was generated to be consistent
with, independent of anything the classifiers look at. The sample is saved to
labels/root_cause_labels.csv so it's a fixed, inspectable eval set rather than being
redrawn every run.
"""
from __future__ import annotations

import re

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from relplatform.generator.postmortems import CATEGORIES

# Ordered rules: first match wins. Deliberately naive (single keyword families) -- this
# is the baseline the AI categorizer has to beat, not a competitive classifier.
KEYWORD_RULES: list[tuple[str, list[str]]] = [
    ("deployment_regression", ["deploy", "rollout", "canary", "regression"]),
    ("configuration_error", ["config", "configuration", "maintenance window"]),
    ("database_issue", ["database", "migration", "query", "index", "lock", "replica"]),
    ("third_party_dependency", ["vendor", "third-party", "upstream provider", "circuit breaker"]),
    ("network_partition", ["network", "dns", "bgp", "availability zone", "packet loss", "security group"]),
    ("traffic_spike", ["traffic spike", "campaign", "surge", "rate limit", "cache hit rate"]),
    ("human_error", ["runbook", "engineer", "manual", "human error", "approver"]),
    ("resource_exhaustion", ["memory", "cpu", "connection pool", "exhaustion", "heap", "autoscaling"]),
]


def keyword_baseline_predict(text: str) -> str:
    lower = text.lower()
    for category, keywords in KEYWORD_RULES:
        if any(kw in lower for kw in keywords):
            return category
    return "deployment_regression"  # most common class fallback


def build_hand_label_sample(incidents: pd.DataFrame, n: int = 100, seed: int = 13) -> pd.DataFrame:
    """Stratified sample across root_cause_category, ~proportional with a floor per
    class so rare categories aren't dropped entirely."""
    rng = pd.Series(range(len(incidents)))
    groups = []
    per_class_floor = max(1, n // (len(CATEGORIES) * 3))
    remaining = n
    df = incidents.sample(frac=1.0, random_state=seed)  # shuffle
    counts_target = {}
    for cat in CATEGORIES:
        cat_n = len(df[df["root_cause_category"] == cat])
        counts_target[cat] = min(cat_n, max(per_class_floor, round(n * cat_n / len(df))))
    total = sum(counts_target.values())
    if total > n:
        scale = n / total
        counts_target = {k: max(1, int(v * scale)) for k, v in counts_target.items()}

    for cat, k in counts_target.items():
        sub = df[df["root_cause_category"] == cat].head(k)
        groups.append(sub)
    sample = pd.concat(groups).drop_duplicates(subset="id")
    if len(sample) > n:
        sample = sample.sample(n=n, random_state=seed)
    return sample.reset_index(drop=True)


def evaluate_keyword_baseline(labeled_sample: pd.DataFrame) -> dict:
    preds = labeled_sample["postmortem_text"].apply(keyword_baseline_predict)
    acc = accuracy_score(labeled_sample["root_cause_category"], preds)
    report = classification_report(labeled_sample["root_cause_category"], preds, zero_division=0, output_dict=True)
    return {"accuracy": float(acc), "n": len(labeled_sample), "report": report, "predictions": preds.tolist()}


def evaluate_ai_categorization(labeled_sample: pd.DataFrame, ai_predictions: list[str]) -> dict:
    acc = accuracy_score(labeled_sample["root_cause_category"], ai_predictions)
    report = classification_report(labeled_sample["root_cause_category"], ai_predictions, zero_division=0, output_dict=True)
    return {"accuracy": float(acc), "n": len(labeled_sample), "report": report, "predictions": ai_predictions}
