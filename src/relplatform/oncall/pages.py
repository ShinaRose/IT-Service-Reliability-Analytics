"""Pages per shift, out-of-hours rate, sleep-hours interruptions, and interrupt-load
concentration. See generator/roster.py's module docstring for why "one incident = one
page" (paging systems coalesce alert storms; alerts themselves are not separately paged).

All percentile-based, not mean-based, per the spec: on-call burden is famously
right-skewed (a handful of brutal shifts, many quiet ones), and a mean hides exactly the
shifts worth knowing about.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from relplatform.oncall.config import OnCallConfig

PAGE_PERCENTILES = [50, 90, 99]


def assign_pages_to_shifts(incidents: pd.DataFrame, shifts: pd.DataFrame) -> pd.DataFrame:
    """Returns `incidents` with `engineer`, `shift_id`, `is_holiday`, `swapped` columns
    added -- the shift whose [shift_start, shift_end) contains this incident's
    started_at. Incidents outside any shift's coverage (shouldn't happen given roster
    generation spans the full simulated window, but not assumed) get nulls."""
    cols = ["engineer", "shift_id", "is_holiday", "swapped"]
    if len(incidents) == 0:
        return incidents.assign(**{c: pd.Series(dtype=object) for c in cols})
    if len(shifts) == 0:
        return incidents.assign(**{c: None for c in cols})

    inc = incidents.sort_values("started_at").reset_index(drop=True)
    sh = shifts.sort_values("shift_start").reset_index(drop=True)

    # suffixes=("", "_shift") keeps the incident's own "id" column unsuffixed and
    # renames only the shift's colliding "id" column, rather than pandas's default
    # id_x/id_y (which silently shadows the incident id if a caller forgets the rename).
    merged = pd.merge_asof(
        inc, sh[["id", "engineer", "shift_start", "shift_end", "is_holiday", "swapped"]],
        left_on="started_at", right_on="shift_start", direction="backward",
        suffixes=("", "_shift"),
    )
    merged = merged.rename(columns={"id_shift": "shift_id"})
    # is_holiday/swapped come back bool-dtyped from the merge; assigning None into a
    # bool column raises TypeError in current pandas even when the boolean mask selects
    # zero rows, so widen to object dtype first -- these columns become effectively
    # nullable (True/False/None) after this point, which is what "no matching shift"
    # actually means.
    for col in ("engineer", "is_holiday", "swapped", "shift_id"):
        merged[col] = merged[col].astype(object)
    in_range = merged["started_at"] < merged["shift_end"]
    merged.loc[~in_range, ["engineer", "is_holiday", "swapped", "shift_id"]] = None
    return merged.drop(columns=["shift_start", "shift_end"])


def pages_per_shift(paged_incidents: pd.DataFrame) -> pd.DataFrame:
    """One row per shift that had at least one page, with its page count."""
    assigned = paged_incidents.dropna(subset=["shift_id"])
    if len(assigned) == 0:
        return pd.DataFrame(columns=["shift_id", "engineer", "n_pages"])
    return (
        assigned.groupby(["shift_id", "engineer"]).size().reset_index(name="n_pages")
        .sort_values("n_pages", ascending=False)
    )


def pages_per_shift_percentiles(paged_incidents: pd.DataFrame, all_shifts: pd.DataFrame) -> dict:
    """Percentiles over ALL shifts, including shifts with zero pages -- a shift that
    paged nobody is a real data point about the distribution's shape, not a missing row
    to drop."""
    if len(all_shifts) == 0:
        return {"n_shifts": 0, "percentiles": {}, "mean": None}

    counts_by_shift = pages_per_shift(paged_incidents).set_index("shift_id")["n_pages"]
    full = all_shifts["id"].map(counts_by_shift).fillna(0).astype(int)

    return {
        "n_shifts": len(full),
        "percentiles": {f"p{p}": float(np.percentile(full, p)) for p in PAGE_PERCENTILES},
        "mean": float(full.mean()),
        "max": int(full.max()),
    }


def out_of_hours_rate(paged_incidents: pd.DataFrame, cfg: OnCallConfig) -> dict:
    assigned = paged_incidents.dropna(subset=["engineer"])
    n = len(assigned)
    if n == 0:
        return {"n_pages": 0, "n_out_of_hours": 0, "out_of_hours_rate": None}
    out_of_hours = ~assigned["started_at"].apply(cfg.is_business_hours)
    return {
        "n_pages": n,
        "n_out_of_hours": int(out_of_hours.sum()),
        "out_of_hours_rate": float(out_of_hours.mean()),
    }


def sleep_hours_interruptions(paged_incidents: pd.DataFrame, cfg: OnCallConfig) -> dict:
    assigned = paged_incidents.dropna(subset=["engineer"])
    n = len(assigned)
    if n == 0:
        return {"n_pages": 0, "n_sleep_hours": 0, "sleep_hours_rate": None}
    in_sleep = assigned["started_at"].apply(cfg.is_sleep_hours)
    return {
        "n_pages": n,
        "n_sleep_hours": int(in_sleep.sum()),
        "sleep_hours_rate": float(in_sleep.mean()),
    }


def gini_coefficient(values: np.ndarray) -> float:
    """Standard Gini coefficient over a non-negative array. 0 = perfectly even load,
    approaching 1 = load concentrated on one person."""
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    total = x.sum()
    if n == 0 or total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * x) - (n + 1) * total) / (n * total))


def interrupt_concentration(paged_incidents: pd.DataFrame) -> dict:
    """Gini coefficient + top-1 engineer's share of total pages -- two views of the same
    question (is paging load spread evenly, or dumped on one or two people)."""
    assigned = paged_incidents.dropna(subset=["engineer"])
    if len(assigned) == 0:
        return {"n_engineers": 0, "gini": 0.0, "top1_share": 0.0, "by_engineer": []}

    by_engineer = assigned.groupby("engineer").size().sort_values(ascending=False)
    total = int(by_engineer.sum())
    top1_share = float(by_engineer.iloc[0] / total) if total else 0.0

    return {
        "n_engineers": len(by_engineer),
        "gini": gini_coefficient(by_engineer.to_numpy()),
        "top1_share": top1_share,
        "by_engineer": [{"engineer": e, "n_pages": int(c)} for e, c in by_engineer.items()],
    }
