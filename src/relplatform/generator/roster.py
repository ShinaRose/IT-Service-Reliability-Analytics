"""Synthetic on-call roster: a single org-wide primary rotation (not per-service --
this platform's 8 services share one paging rotation, the common setup for a team this
size), weekly shifts, with realistic handover jitter and holiday coverage swaps.

Assumption surfaced here rather than left implicit: every incident is modeled as one
page to whichever engineer's shift covers its started_at. The dozens of alerts in that
incident's storm are NOT modeled as separate pages -- real paging tools (PagerDuty,
Opsgenie) coalesce an alert storm into one incident notification, which is exactly what
relplatform.analytics.clustering's alert dedup already recovers computationally.
"""
from __future__ import annotations

import random
from datetime import date, datetime, timedelta

ENGINEERS = [f"oncall-eng-{i}" for i in range(1, 7)]  # 6-person rotation

SHIFT_DAYS = 7
HANDOVER_JITTER_MINUTES = 20
HOLIDAY_SWAP_PROBABILITY = 0.3  # chance a holiday-covering shift trades to the next engineer


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """weekday: Monday=0..Sunday=6. n: 1-indexed occurrence within the month."""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    d += timedelta(days=offset + 7 * (n - 1))
    return d


def company_holidays(start: datetime, end: datetime) -> set[date]:
    """A small set of fixed/floating-date company holidays, illustrative only -- not
    tied to any specific jurisdiction's actual public holiday calendar."""
    holidays: set[date] = set()
    for year in range(start.year, end.year + 1):
        candidates = [
            date(year, 1, 1),                              # New Year's Day
            date(year, 7, 4),                                # Mid-year holiday
            _nth_weekday_of_month(year, 11, 3, 4),           # Late-autumn holiday (4th Thursday of Nov)
            date(year, 12, 25),                              # Winter holiday
        ]
        holidays.update(d for d in candidates if start.date() <= d <= end.date())
    return holidays


def generate_roster(
    rng: random.Random, start: datetime, end: datetime,
    engineers: list[str] = ENGINEERS, shift_days: int = SHIFT_DAYS,
) -> list[dict]:
    """Weekly rotation covering [start, end). Each shift's boundary carries a small
    random handover jitter (the actual handoff rarely happens at the exact clean
    boundary) that compounds shift-to-shift -- a realistic property of real rotations,
    not a bug. Shifts overlapping a company holiday are flagged `is_holiday`, and some
    fraction of those trade to the next engineer in rotation (`swapped`), modeling a
    volunteer covering so the nominal owner gets the day off.
    """
    holidays = company_holidays(start, end)
    shifts: list[dict] = []
    cursor = start
    i = 0
    shift_counter = 0

    while cursor < end:
        shift_counter += 1
        engineer = engineers[i % len(engineers)]
        nominal_end = min(cursor + timedelta(days=shift_days), end)
        is_last = nominal_end >= end
        jitter = timedelta(0) if is_last else timedelta(minutes=rng.randint(-HANDOVER_JITTER_MINUTES, HANDOVER_JITTER_MINUTES))
        shift_end = max(cursor + timedelta(hours=1), nominal_end + jitter)  # never collapse a shift to <1h
        shift_end = min(shift_end, end)

        is_holiday = any(cursor.date() <= h <= shift_end.date() for h in holidays)
        swapped = False
        if is_holiday and rng.random() < HOLIDAY_SWAP_PROBABILITY:
            engineer = engineers[(i + 1) % len(engineers)]
            swapped = True

        shifts.append(dict(
            id=f"SHIFT-{shift_counter:04d}", engineer=engineer,
            shift_start=cursor, shift_end=shift_end,
            is_holiday=is_holiday, swapped=swapped,
        ))
        cursor = shift_end
        i += 1

    return shifts
