"""Numeric extraction + membership check used to guarantee an exec summary never states
a metric value that isn't present in the input it was given. This is the mechanism
behind the mandatory hallucination test (relplatform.eval.hallucination_check /
tests/test_hallucination.py) -- it is intentionally NOT an LLM check: catching a
fabricated number is a string/number problem, not a judgment call.
"""
from __future__ import annotations

import re

_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# (?<!\d) before the optional sign: without it, a range like "3-5%" was parsed as the
# minus sign belonging to "5" (tokens 3.0, -5.0 instead of 3.0, 5.0) because "-?" happily
# consumed a hyphen that was actually a range separator, not a leading minus.
_NUM_RE = re.compile(r"(?<!\d)-?\d[\d,]*(?:\.\d+)?%?")


def extract_numbers(text: str) -> list[float]:
    # Strip ISO dates first so their day/month components (e.g. the "02" in
    # "2026-11-02") never get treated as standalone metric values -- a fabricated "2.1%"
    # shouldn't be able to hide behind a date fragment that happens to round to 2.
    text = _ISO_DATE_RE.sub(" ", text)
    out = []
    for m in _NUM_RE.finditer(text):
        tok = m.group(0).replace(",", "").rstrip("%")
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out


def _close_enough(n: float, allowed: float) -> bool:
    if round(n, 1) == round(allowed, 1):
        return True
    # Whole-number reformatting ("7%" for an input of "7.29%") wasn't actually covered by
    # either rule above or below: round(7.0,1) != round(7.29,1), and 0.29 exceeds the 3%
    # relative tolerance below. Only fires when `n` itself looks like a rounded whole
    # number (no fractional part) -- unlike a bare `round(n) == round(allowed)` check
    # (which was tried and reverted), this doesn't reintroduce the 1.8-vs-2.1 collision:
    # 2.1 has a fractional part, so this branch never applies to it.
    if n == int(n) and round(allowed) == n:
        return True
    # Small relative slack so the model can reformat "86.4%" as "86%" without being
    # flagged, but deliberately no generous absolute floor -- that's what let two
    # genuinely different small numbers (e.g. 1.8 and 2.1) both round to the same nearby
    # integer and collide.
    tol = max(0.05, 0.03 * abs(allowed))
    return abs(n - allowed) <= tol


def find_unsupported_numbers(output_text: str, context_text: str) -> list[float]:
    """Returns the list of numbers in `output_text` that have no reasonably-close match
    in `context_text` (tolerant of rounding, e.g. model writing "7%" for an input value
    of "7.29%"). A non-empty return means the output likely hallucinated a figure."""
    allowed = extract_numbers(context_text)
    found = extract_numbers(output_text)
    unsupported = [n for n in found if not any(_close_enough(n, a) for a in allowed)]
    return unsupported
