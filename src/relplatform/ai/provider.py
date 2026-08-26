"""ModelProvider interface + implementations, selected by env var.

Every provider gets generate(), embed(), and structured_output() for free from the base
class -- concrete providers only implement `_generate_raw`. embed() always routes to the
local sentence-transformers model (relplatform.analytics.embeddings) regardless of which
generation provider is active: embeddings are zero-budget and local by design, generation
is the only thing that varies by provider.

structured_output() enforces a JSON schema with jsonschema.validate and retries with the
validation error fed back into the prompt on failure -- this is what lets the AI tasks
(root-cause categorization, narrative summaries) return guaranteed-parseable JSON instead
of "usually valid" JSON.

Selection: RELPLATFORM_PROVIDER env var = "ollama" (default) | "gemini" | "mock".
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jsonschema
import numpy as np

from relplatform.config import GEMINI_API_KEY, GEMINI_MODEL, MODEL_PROVIDER, OLLAMA_HOST, OLLAMA_MODEL

# Neither ollama.Client nor Gemini's generate_content default to a bounded timeout --
# ollama.Client(timeout=None) (the implicit default) passes straight through to
# httpx.Client(timeout=None), which is genuinely unbounded, not "use httpx's usual
# default." A single hung call blocks forever. 60s is generous relative to this
# project's observed real-world median call latency (~12.5s), so it only ever cuts off
# calls that are already pathological, not slow-but-fine ones.
REQUEST_TIMEOUT_SECONDS = 60.0


@dataclass
class GenerateResult:
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    provider: str
    model: str


@dataclass
class StructuredResult:
    data: dict
    tokens_in: int
    tokens_out: int
    latency_ms: float
    retries: int
    provider: str
    model: str
    valid: bool = True
    raw_text: str = ""


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) used when a provider doesn't report real
    usage counts (e.g. Ollama's non-streaming API omits it in some versions)."""
    return max(1, len(text) // 4)


class ModelProvider(ABC):
    name: str = "base"

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def _generate_raw(self, prompt: str, system: str | None, max_tokens: int, temperature: float) -> tuple[str, int, int]:
        """Returns (text, tokens_in, tokens_out)."""
        raise NotImplementedError

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 512, temperature: float = 0.2) -> GenerateResult:
        t0 = time.perf_counter()
        text, tin, tout = self._generate_raw(prompt, system, max_tokens, temperature)
        latency_ms = (time.perf_counter() - t0) * 1000
        return GenerateResult(text=text, tokens_in=tin, tokens_out=tout, latency_ms=latency_ms, provider=self.name, model=self.model)

    def embed(self, con, texts: list[str]) -> np.ndarray:
        from relplatform.analytics.embeddings import embed_texts

        return embed_texts(con, texts)

    def structured_output(
        self, prompt: str, schema: dict, system: str | None = None, max_retries: int = 3, max_tokens: int = 768,
    ) -> StructuredResult:
        instructions = (
            (system or "") + "\n\nRespond with ONLY a single JSON object matching this schema "
            "(no prose, no markdown fences).\nJSON Schema:\n" + json.dumps(schema)
        )
        last_error = None
        total_latency = 0.0
        total_tin = total_tout = 0
        raw_text = ""

        for attempt in range(max_retries + 1):
            p = prompt if attempt == 0 else f"{prompt}\n\nYour previous response was invalid: {last_error}\nReturn corrected JSON only."
            result = self.generate(p, system=instructions, max_tokens=max_tokens, temperature=0.1)
            total_latency += result.latency_ms
            total_tin += result.tokens_in
            total_tout += result.tokens_out
            raw_text = result.text

            data, err = _parse_and_validate(result.text, schema)
            if err is None:
                return StructuredResult(
                    data=data, tokens_in=total_tin, tokens_out=total_tout, latency_ms=total_latency,
                    retries=attempt, provider=self.name, model=self.model, valid=True, raw_text=raw_text,
                )
            last_error = err

        return StructuredResult(
            data={}, tokens_in=total_tin, tokens_out=total_tout, latency_ms=total_latency,
            retries=max_retries, provider=self.name, model=self.model, valid=False, raw_text=raw_text,
        )


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return text
    return text[start : end + 1]


def _parse_and_validate(text: str, schema: dict) -> tuple[dict, str | None]:
    try:
        data = json.loads(_extract_json(text))
    except json.JSONDecodeError as e:
        return {}, f"invalid JSON: {e}"
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        return {}, f"schema violation: {e.message}"
    return data, None


class OllamaProvider(ModelProvider):
    name = "ollama"

    def __init__(self, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST, timeout: float = REQUEST_TIMEOUT_SECONDS):
        super().__init__(model)
        import ollama

        self._client = ollama.Client(host=host, timeout=timeout)

    def _generate_raw(self, prompt, system, max_tokens, temperature):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat(
            model=self.model, messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        text = resp["message"]["content"]
        tin = resp.get("prompt_eval_count") or _estimate_tokens(prompt + (system or ""))
        tout = resp.get("eval_count") or _estimate_tokens(text)
        return text, tin, tout


class GeminiProvider(ModelProvider):
    name = "gemini"

    def __init__(self, model: str = GEMINI_MODEL, api_key: str = GEMINI_API_KEY):
        super().__init__(model)
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._genai = genai

    def _generate_raw(self, prompt, system, max_tokens, temperature):
        model = self._genai.GenerativeModel(self.model, system_instruction=system)
        resp = model.generate_content(
            prompt,
            generation_config=self._genai.types.GenerationConfig(
                max_output_tokens=max_tokens, temperature=temperature,
            ),
            request_options={"timeout": REQUEST_TIMEOUT_SECONDS},
        )
        text = resp.text
        usage = getattr(resp, "usage_metadata", None)
        tin = getattr(usage, "prompt_token_count", None) or _estimate_tokens(prompt + (system or ""))
        tout = getattr(usage, "candidates_token_count", None) or _estimate_tokens(text)
        return text, tin, tout


_SCHEMA_MARKER = "JSON Schema:\n"


def _extract_schema_from_system(system: str | None) -> dict | None:
    """MockProvider has no real model to read the schema instructions, so it recovers the
    schema `structured_output()` embedded into the system prompt and synthesizes a
    conforming dummy response -- this keeps the base class's real parse+validate(+retry)
    path exercised end-to-end even in tests/offline mode, rather than special-casing
    structured_output() away."""
    if not system or _SCHEMA_MARKER not in system:
        return None
    try:
        return json.loads(system.split(_SCHEMA_MARKER, 1)[1])
    except json.JSONDecodeError:
        return None


class MockProvider(ModelProvider):
    """Deterministic, offline provider for tests and CI. `fixtures` maps a substring of
    the prompt (checked in insertion order) to a canned response string. If no fixture
    matches and the request is a structured_output() call, a schema-conforming dummy
    object is synthesized so the real base-class validation path still succeeds."""

    name = "mock"

    def __init__(self, model: str = "mock-1", fixtures: dict[str, str] | None = None):
        super().__init__(model)
        self.fixtures = fixtures or {}
        self.calls: list[str] = []

    def _generate_raw(self, prompt, system, max_tokens, temperature):
        self.calls.append(prompt)
        for key, response in self.fixtures.items():
            if key in prompt:
                return response, _estimate_tokens(prompt), _estimate_tokens(response)

        schema = _extract_schema_from_system(system)
        if schema is not None:
            text = json.dumps(_synthesize_from_schema(schema))
        else:
            text = f"[mock response for: {prompt[:60]}...]"
        return text, _estimate_tokens(prompt), _estimate_tokens(text)


def _synthesize_from_schema(schema: dict) -> Any:
    t = schema.get("type", "object")
    if t == "object":
        out = {}
        props = schema.get("properties", {})
        required = schema.get("required", list(props.keys()))
        for key in required:
            out[key] = _synthesize_from_schema(props.get(key, {"type": "string"}))
        return out
    if t == "array":
        item_schema = schema.get("items", {"type": "string"})
        n = schema.get("minItems", 1)
        return [_synthesize_from_schema(item_schema) for _ in range(max(1, n))]
    if t == "string":
        if "enum" in schema:
            return schema["enum"][0]
        return "mock_value"
    if t == "number":
        return 0.5
    if t == "integer":
        return 1
    if t == "boolean":
        return True
    return None


def get_provider(provider_name: str | None = None) -> ModelProvider:
    name = (provider_name or MODEL_PROVIDER).lower()
    if name == "ollama":
        return OllamaProvider()
    if name == "gemini":
        return GeminiProvider()
    if name == "mock":
        return MockProvider()
    raise ValueError(f"Unknown provider: {name}")
