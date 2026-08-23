"""Change-failure model: logistic regression on deploy features, estimating the
probability that a deployment causes an incident, so risky changes can be flagged
before merge.

Deliberately a plain, inspectable logistic regression (not a black box) -- the whole
point of pre-merge risk flagging is that an engineer can be told *why* a change is
risky. Features are all things known at merge time: change size, commit count, timing,
and the target service (different services carry different latent risk, e.g. a
payments change is riskier than a notification change of the same size).

Label: `caused_incident` from `relplatform.analytics.dora.label_deploy_caused_incidents`
(a time-proximity heuristic), NOT the generator's ground-truth `triggering_deploy_id` --
this mirrors what a real system would have to work with. The eval suite separately
scores this heuristic+model pipeline against the ground truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

NUMERIC_FEATURES = ["log_lines_changed", "commit_count", "lead_time_hours", "is_weekend", "is_off_hours"]
CATEGORICAL_FEATURES = ["service"]


def _prep_features(deployments: pd.DataFrame) -> pd.DataFrame:
    df = deployments.copy()
    df["log_lines_changed"] = np.log1p(df["lines_changed"])
    df["is_weekend"] = df["is_weekend"].astype(int)
    df["is_off_hours"] = df["is_off_hours"].astype(int)
    return df


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    # C=0.3 (moderate L2 shrinkage): with a ~7% base rate and one-hot service dummies,
    # an unregularized fit lets low-volume services (e.g. recommendation-service, <1
    # deploy/week) swing on a handful of samples; shrinkage stabilizes cross-validated AUC.
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.3)
    return Pipeline([("pre", pre), ("clf", clf)])


def time_split(deployments: pd.DataFrame, test_frac: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = deployments.sort_values("deployed_at")
    if len(df) == 0:
        return df, df
    # Clamped: test_frac=0 (or a small enough df) made cutoff_idx == len(df), one past the
    # last row, and df.iloc[cutoff_idx] raised IndexError instead of returning "everything
    # in train, nothing in test".
    cutoff_idx = min(int(len(df) * (1 - test_frac)), len(df) - 1)
    cutoff = df.iloc[cutoff_idx]["deployed_at"]
    return df[df["deployed_at"] < cutoff], df[df["deployed_at"] >= cutoff]


def train_change_failure_model(deployments_labeled: pd.DataFrame) -> dict:
    df = _prep_features(deployments_labeled)
    train, test = time_split(df)

    pipe = build_pipeline()
    pipe.fit(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train["caused_incident"])

    proba_test = pipe.predict_proba(test[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    y_test = test["caused_incident"].values

    metrics = {"n_train": len(train), "n_test": len(test), "positive_rate_test": float(y_test.mean())}

    # Headline metric: 5-fold stratified CV AUC over the whole dataset. At a ~7% base
    # rate, a single chronological 75/25 holdout has too few positives (~70-90) for a
    # stable AUC estimate; CV averages over 5 such estimates instead of trusting one.
    full = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_full = df["caused_incident"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cv_scores = cross_val_score(pipe, full, y_full, cv=skf, scoring="roc_auc")
    metrics["cv_roc_auc_mean"] = float(cv_scores.mean())
    metrics["cv_roc_auc_std"] = float(cv_scores.std())

    if y_test.sum() > 0 and y_test.sum() < len(y_test):
        metrics["holdout_roc_auc"] = float(roc_auc_score(y_test, proba_test))
        metrics["average_precision"] = float(average_precision_score(y_test, proba_test))
        preds = (proba_test >= 0.5).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
        metrics["precision_at_0.5"] = float(p)
        metrics["recall_at_0.5"] = float(r)
        metrics["f1_at_0.5"] = float(f1)

    # coefficients, for "why is this risky" explanations
    ohe = pipe.named_steps["pre"].named_transformers_["cat"]
    feature_names = NUMERIC_FEATURES + list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    coefs = pipe.named_steps["clf"].coef_[0]
    metrics["coefficients"] = dict(sorted(zip(feature_names, coefs.tolist()), key=lambda kv: -abs(kv[1])))

    return {"pipeline": pipe, "metrics": metrics, "test_scores": pd.DataFrame({
        "deployment_id": test["id"].values, "service": test["service"].values,
        "risk_probability": proba_test, "actual_caused_incident": y_test,
    })}


def score_deploys(pipe: Pipeline, deployments: pd.DataFrame, high_risk_percentile: float = 0.90) -> pd.DataFrame:
    """Pre-merge risk flagging: score any deploy batch and flag the top decile as high-risk."""
    df = _prep_features(deployments)
    proba = pipe.predict_proba(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    out = deployments.copy()
    out["risk_probability"] = proba
    threshold = np.quantile(proba, high_risk_percentile) if len(proba) > 1 else 0.5
    out["flagged_high_risk"] = out["risk_probability"] >= threshold
    return out
