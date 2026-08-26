from relplatform.ai.provider import REQUEST_TIMEOUT_SECONDS, MockProvider, OllamaProvider
from relplatform.ai.schemas import ROOT_CAUSE_SCHEMA


def test_mock_generate_basic():
    provider = MockProvider(fixtures={"hello": "hi there"})
    result = provider.generate("hello world")
    assert result.text == "hi there"
    assert result.tokens_in > 0 and result.tokens_out > 0
    assert result.provider == "mock"


def test_mock_structured_output_synthesizes_valid_schema():
    provider = MockProvider()
    result = provider.structured_output("categorize this postmortem", ROOT_CAUSE_SCHEMA)
    assert result.valid
    assert result.data["root_cause_category"] in ROOT_CAUSE_SCHEMA["properties"]["root_cause_category"]["enum"]
    assert isinstance(result.data["contributing_factors"], list)


def test_mock_structured_output_uses_fixture_when_valid():
    fixture_json = '{"root_cause_category": "database_issue", "contributing_factors": ["missing index"], "preventive_action": "add index"}'
    provider = MockProvider(fixtures={"INC-999": fixture_json})
    result = provider.structured_output("Postmortem for INC-999:\n...", ROOT_CAUSE_SCHEMA)
    assert result.data["root_cause_category"] == "database_issue"


def test_structured_output_retries_and_fails_gracefully():
    class BrokenProvider(MockProvider):
        def _generate_raw(self, prompt, system, max_tokens, temperature):
            return "not json at all", 5, 5

    provider = BrokenProvider()
    result = provider.structured_output("categorize", ROOT_CAUSE_SCHEMA, max_retries=2)
    assert result.valid is False
    assert result.retries == 2


def test_embed_returns_expected_shape(memdb):
    provider = MockProvider()
    vectors = provider.embed(memdb, ["hello world", "another message"])
    assert vectors.shape[0] == 2
    assert vectors.shape[1] > 0


def test_ollama_provider_has_a_bounded_timeout():
    # A single hung call used to block forever: ollama.Client(host=host) with no
    # timeout kwarg passes timeout=None straight through to httpx.Client, which means
    # genuinely unbounded, not "httpx's usual default." This constructs the real
    # ollama.Client (no network I/O happens in __init__, just httpx.Client setup) and
    # inspects its configured timeout directly, rather than mocking the SDK.
    provider = OllamaProvider(host="http://localhost:11434")
    assert provider._client._client.timeout.connect == REQUEST_TIMEOUT_SECONDS


def test_ollama_provider_respects_custom_timeout():
    provider = OllamaProvider(host="http://localhost:11434", timeout=5.0)
    assert provider._client._client.timeout.connect == 5.0
