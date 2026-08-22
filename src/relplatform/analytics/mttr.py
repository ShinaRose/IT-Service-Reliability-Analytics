"""Per-service MTTR distribution fitting.

Restore-time data is right-skewed with a heavy tail (most incidents resolve quickly, a
few take much longer) -- reporting a mean is misleading. We fit both log-normal and
Weibull (common choices for time-to-repair data in reliability engineering), pick the
better fit per service by AIC, and report percentiles from the fitted distribution
rather than the mean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PERCENTILES = [50, 75, 90, 95, 99]


def _fit_lognorm(x: np.ndarray) -> tuple[dict, float]:
    shape, loc, scale = stats.lognorm.fit(x, floc=0)
    ll = np.sum(stats.lognorm.logpdf(x, shape, loc=loc, scale=scale))
    aic = 2 * 2 - 2 * ll
    return {"dist": "lognorm", "shape": shape, "loc": loc, "scale": scale}, aic


def _fit_weibull(x: np.ndarray) -> tuple[dict, float]:
    shape, loc, scale = stats.weibull_min.fit(x, floc=0)
    ll = np.sum(stats.weibull_min.logpdf(x, shape, loc=loc, scale=scale))
    aic = 2 * 2 - 2 * ll
    return {"dist": "weibull_min", "shape": shape, "loc": loc, "scale": scale}, aic


def _percentiles_from_fit(params: dict) -> dict:
    if params["dist"] == "lognorm":
        dist = stats.lognorm(params["shape"], loc=params["loc"], scale=params["scale"])
    else:
        dist = stats.weibull_min(params["shape"], loc=params["loc"], scale=params["scale"])
    return {f"p{p}": float(dist.ppf(p / 100)) for p in PERCENTILES}


def fit_mttr_per_service(incidents: pd.DataFrame, min_n: int = 8) -> dict:
    df = incidents.copy()
    df["restore_minutes"] = (
        pd.to_datetime(df["resolved_at"]) - pd.to_datetime(df["started_at"])
    ).dt.total_seconds() / 60
    df = df[df["restore_minutes"] > 0]

    results = {}
    for service, grp in df.groupby("service"):
        x = grp["restore_minutes"].values
        empirical = {f"p{p}": float(np.percentile(x, p)) for p in PERCENTILES}
        if len(x) < min_n:
            results[service] = {"n": len(x), "fit": None, "empirical_percentiles": empirical,
                                 "note": f"n<{min_n}, reporting empirical percentiles only"}
            continue

        candidates = []
        for fit_fn in (_fit_lognorm, _fit_weibull):
            try:
                params, aic = fit_fn(x)
                candidates.append((aic, params))
            except Exception:
                continue
        candidates.sort(key=lambda c: c[0])
        best_aic, best_params = candidates[0]
        ks_stat, ks_p = stats.kstest(x, best_params["dist"],
                                      args=(best_params["shape"], best_params["loc"], best_params["scale"]))

        results[service] = {
            "n": len(x),
            "fit": {"distribution": best_params["dist"], "shape": round(best_params["shape"], 4),
                    "scale": round(best_params["scale"], 4), "aic": round(best_aic, 2),
                    "ks_statistic": round(float(ks_stat), 4), "ks_pvalue": round(float(ks_p), 4)},
            "fitted_percentiles_minutes": {k: round(v, 1) for k, v in _percentiles_from_fit(best_params).items()},
            "empirical_percentiles_minutes": {k: round(v, 1) for k, v in empirical.items()},
        }
    return results
