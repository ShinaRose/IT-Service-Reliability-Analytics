"""Content-hash cache for AI task results, backed by DuckDB's `ai_cache` table.

Batch jobs are the only way AI tasks run (per the zero-budget constraint: no live,
per-request inference), so every task result is cached by sha256(task, provider,
model, input) and reused across runs. Re-running a batch job after adding new alerts
or incidents only pays for the new content.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


def content_hash(task: str, provider: str, model: str, input_text: str) -> str:
    h = hashlib.sha256()
    h.update("|".join([task, provider, model, input_text]).encode("utf-8"))
    return h.hexdigest()


@dataclass
class CachedCallStats:
    hit: bool
    tokens_in: int
    tokens_out: int
    latency_ms: float


def get_cached(con, key: str) -> dict | None:
    row = con.execute("SELECT output_json, tokens_in, tokens_out, latency_ms FROM ai_cache WHERE content_hash = ?", [key]).fetchone()
    if row is None:
        return None
    output_json, tin, tout, latency = row
    return {"output": json.loads(output_json), "tokens_in": tin, "tokens_out": tout, "latency_ms": latency, "hit": True}


def put_cached(con, key: str, task: str, provider: str, model: str, input_text: str, output: dict,
               tokens_in: int, tokens_out: int, latency_ms: float) -> None:
    con.execute(
        "INSERT OR REPLACE INTO ai_cache (content_hash, task, provider, model, input_text, output_json, "
        "tokens_in, tokens_out, latency_ms) VALUES (?,?,?,?,?,?,?,?,?)",
        [key, task, provider, model, input_text, json.dumps(output), tokens_in, tokens_out, latency_ms],
    )


def cached_structured_call(con, provider, task: str, prompt: str, schema: dict, system: str | None = None,
                            max_retries: int = 3) -> tuple[dict, CachedCallStats]:
    """structured_output() wrapped with the content-hash cache. Returns (data, stats)."""
    key = content_hash(task, provider.name, provider.model, prompt)
    cached = get_cached(con, key)
    if cached is not None:
        return cached["output"], CachedCallStats(hit=True, tokens_in=cached["tokens_in"], tokens_out=cached["tokens_out"], latency_ms=cached["latency_ms"])

    result = provider.structured_output(prompt, schema, system=system, max_retries=max_retries)
    put_cached(con, key, task, provider.name, provider.model, prompt, result.data, result.tokens_in, result.tokens_out, result.latency_ms)
    return result.data, CachedCallStats(hit=False, tokens_in=result.tokens_in, tokens_out=result.tokens_out, latency_ms=result.latency_ms)


def cached_generate_call(con, provider, task: str, prompt: str, system: str | None = None,
                          max_tokens: int = 512) -> tuple[str, CachedCallStats]:
    key = content_hash(task, provider.name, provider.model, prompt)
    cached = get_cached(con, key)
    if cached is not None:
        return cached["output"]["text"], CachedCallStats(hit=True, tokens_in=cached["tokens_in"], tokens_out=cached["tokens_out"], latency_ms=cached["latency_ms"])

    result = provider.generate(prompt, system=system, max_tokens=max_tokens)
    put_cached(con, key, task, provider.name, provider.model, prompt, {"text": result.text}, result.tokens_in, result.tokens_out, result.latency_ms)
    return result.text, CachedCallStats(hit=False, tokens_in=result.tokens_in, tokens_out=result.tokens_out, latency_ms=result.latency_ms)
