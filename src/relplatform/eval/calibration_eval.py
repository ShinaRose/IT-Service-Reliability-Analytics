"""Change-failure model calibration curve + PR-AUC vs. a no-skill baseline.

average_precision_score (PR-AUC) is already computed in
relplatform.analytics.change_failure.train_change_failure_model -- this module adds the
one comparison point that number needs to mean anything: a no-skill/random classifier's
expected PR-AUC equals the positive class's prevalence, so "0.34 PR-AUC" is only good
news if the prevalence is well below 0.34, and bad news if it's above. It also adds the
calibration curve itself (predicted probability vs. observed frequency, in quantile
bins) -- is a deploy the model calls "70% risky" actually risky 70% of the time, or is
the model just confidently wrong.
"""
from __future__ import annotations

import pandas as pd
from sklearn.calibration import calibration_curve

from relplatform.analytics.change_failure import train_change_failure_model


def evaluate_change_failure_calibration(deployments_labeled: pd.DataFrame, n_bins: int = 10) -> dict:
    result = train_change_failure_model(deployments_labeled)
    metrics = result["metrics"]
    test_scores = result["test_scores"]

    y_test = test_scores["actual_caused_incident"].to_numpy()
    y_prob = test_scores["risk_probability"].to_numpy()

    out = {
        "n_test": int(len(test_scores)),
        "positive_rate_test": metrics.get("positive_rate_test"),
        "holdout_roc_auc": metrics.get("holdout_roc_auc"),
        "average_precision": metrics.get("average_precision"),
        # A no-skill classifier's expected PR-AUC equals the base rate -- the "vs
        # baseline" comparison point the spec asks for, not a second model to train.
        "baseline_average_precision": metrics.get("positive_rate_test"),
        "calibration_curve": None,
    }

    if 0 < y_test.sum() < len(y_test):
        n_bins_eff = max(2, min(n_bins, int(y_test.sum())))
        try:
            prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=n_bins_eff, strategy="quantile")
            out["calibration_curve"] = {
                "predicted_probability": prob_pred.tolist(),
                "observed_frequency": prob_true.tolist(),
                "n_bins": n_bins_eff,
            }
        except ValueError as e:
            out["calibration_curve_error"] = str(e)

    return out
