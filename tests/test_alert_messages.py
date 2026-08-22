import random
import re
from collections import Counter

from relplatform.generator.alert_messages import TEMPLATES, sample_alert


def _build_matcher(template: str) -> re.Pattern:
    pattern = re.escape(template)
    pattern = pattern.replace(re.escape("{service}"), ".+?").replace(re.escape("{val}"), ".+?")
    return re.compile("^" + pattern + "$")


_MATCHERS = {signal: [_build_matcher(t) for t in templates] for signal, templates in TEMPLATES.items()}


def _signal_of(message: str) -> str:
    for signal, matchers in _MATCHERS.items():
        if any(m.match(message) for m in matchers):
            return signal
    return "unknown"


def test_category_biases_signal_distribution():
    rng = random.Random(1)
    resource_signals = Counter()
    for _ in range(400):
        msg, _ = sample_alert(rng, "payments-service", storm_intensity=0.5, category="resource_exhaustion")
        resource_signals[_signal_of(msg)] += 1

    preferred = {"cpu", "memory", "connection_pool", "queue_depth"}
    preferred_share = sum(resource_signals[s] for s in preferred) / sum(resource_signals.values())
    assert preferred_share > 0.5  # clearly dominant, not uniform-random (uniform would be ~4/9=0.44)


def test_no_category_falls_back_to_uniform_ish():
    rng = random.Random(2)
    signals = Counter()
    for _ in range(900):
        msg, _ = sample_alert(rng, "auth-service", storm_intensity=0.3, category=None)
        signals[_signal_of(msg)] += 1
    # every signal type should show up at least a little with no category bias
    assert len(signals) == len(TEMPLATES)
