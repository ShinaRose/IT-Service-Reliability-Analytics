"""Aggregates per-call latency/token stats from a batch job into a report. Mandatory
per the eval spec: every AI batch run should be able to say what it cost."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from relplatform.ai.cache import CachedCallStats


@dataclass
class BatchReport:
    task: str
    n_items: int
    n_cache_hits: int
    n_cache_misses: int
    total_tokens_in: int
    total_tokens_out: int
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_mean: float
    wall_clock_s: float

    def to_dict(self) -> dict:
        return {
            "task": self.task, "n_items": self.n_items, "n_cache_hits": self.n_cache_hits,
            "n_cache_misses": self.n_cache_misses, "cache_hit_rate": round(self.n_cache_hits / max(1, self.n_items), 3),
            "total_tokens_in": self.total_tokens_in, "total_tokens_out": self.total_tokens_out,
            "total_tokens": self.total_tokens_in + self.total_tokens_out,
            "latency_ms_p50": round(self.latency_ms_p50, 1), "latency_ms_p95": round(self.latency_ms_p95, 1),
            "latency_ms_mean": round(self.latency_ms_mean, 1), "wall_clock_s": round(self.wall_clock_s, 2),
        }


def build_batch_report(task: str, stats: list[CachedCallStats], wall_clock_s: float) -> BatchReport:
    latencies = np.array([s.latency_ms for s in stats]) if stats else np.array([0.0])
    return BatchReport(
        task=task,
        n_items=len(stats),
        n_cache_hits=sum(1 for s in stats if s.hit),
        n_cache_misses=sum(1 for s in stats if not s.hit),
        total_tokens_in=sum(s.tokens_in for s in stats),
        total_tokens_out=sum(s.tokens_out for s in stats),
        latency_ms_p50=float(np.percentile(latencies, 50)),
        latency_ms_p95=float(np.percentile(latencies, 95)),
        latency_ms_mean=float(np.mean(latencies)),
        wall_clock_s=wall_clock_s,
    )
