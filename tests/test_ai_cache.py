from relplatform.ai.cache import cached_generate_call, cached_structured_call
from relplatform.ai.provider import MockProvider
from relplatform.ai.schemas import ROOT_CAUSE_SCHEMA


def test_cached_structured_call_hits_on_repeat(memdb):
    provider = MockProvider()
    data1, stats1 = cached_structured_call(memdb, provider, "root_cause", "postmortem A", ROOT_CAUSE_SCHEMA)
    assert stats1.hit is False
    assert len(provider.calls) == 1

    data2, stats2 = cached_structured_call(memdb, provider, "root_cause", "postmortem A", ROOT_CAUSE_SCHEMA)
    assert stats2.hit is True
    assert len(provider.calls) == 1  # no new call made
    assert data1 == data2


def test_cache_keyed_by_content_not_just_task(memdb):
    provider = MockProvider()
    _, stats_a = cached_structured_call(memdb, provider, "root_cause", "postmortem A", ROOT_CAUSE_SCHEMA)
    _, stats_b = cached_structured_call(memdb, provider, "root_cause", "postmortem B", ROOT_CAUSE_SCHEMA)
    assert stats_a.hit is False
    assert stats_b.hit is False
    assert len(provider.calls) == 2


def test_cached_generate_call_hits_on_repeat(memdb):
    provider = MockProvider(fixtures={"summarize": "a short narrative"})
    text1, stats1 = cached_generate_call(memdb, provider, "narrative", "summarize this cluster")
    text2, stats2 = cached_generate_call(memdb, provider, "narrative", "summarize this cluster")
    assert text1 == text2 == "a short narrative"
    assert stats1.hit is False
    assert stats2.hit is True
