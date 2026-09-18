"""Shared scaffolding for the campaign agent — every stage v0..v6 uses this.

The agent's job, in one sentence:
    "Given a brief, gather context with tools, plan N looks with an LLM,
     render each look on one of three image backends, QA the output, and
     report progress."
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backends import TIME_SCALE, ImageProvider, ImageResult, PlannerLLM, ToolBox, make_providers, sim, sim_sleep  # noqa: E402
from backends.mock_providers import PlannedJob  # noqa: E402

DEFAULT_N = 40
BRIEF = "Autumn editorial campaign, oversized tailoring, studio light"


@dataclass
class Metrics:
    stage: str
    description: str
    n_jobs: int
    succeeded: int = 0
    failed: int = 0
    rate_limited: int = 0
    crashed: bool = False
    crash_reason: str = ""
    wall_real_s: float = 0.0
    wall_sim_s: float = 0.0
    time_to_first_result_sim_s: float | None = None
    peak_mem_mb: float = 0.0
    provider_peak_in_flight: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def summary(self) -> str:
        ttfr = "—" if self.time_to_first_result_sim_s is None else f"{self.time_to_first_result_sim_s:6.1f}s"
        status = f"CRASHED@{self.succeeded}" if self.crashed else f"{self.succeeded:3d}/{self.n_jobs}"
        return (
            f"{self.stage:<4} {self.description:<34} "
            f"wall={self.wall_sim_s:7.1f}s(sim) {self.wall_real_s:5.1f}s(real)  "
            f"ok={status:<8} 429s={self.rate_limited:<4} "
            f"peak_mem={self.peak_mem_mb:6.1f}MB  first_result={ttfr}"
        )


class Run:
    """Context manager that measures wall time and peak memory for one stage."""

    def __init__(self, stage: str, description: str, n_jobs: int, providers: dict[str, ImageProvider]):
        self.m = Metrics(stage, description, n_jobs)
        self.providers = providers
        self._t0 = 0.0

    def __enter__(self) -> "Run":
        for p in self.providers.values():
            p.reset()
        tracemalloc.start()
        self._t0 = time.perf_counter()
        return self

    def mark_first_result(self) -> None:
        if self.m.time_to_first_result_sim_s is None:
            self.m.time_to_first_result_sim_s = (time.perf_counter() - self._t0) * TIME_SCALE

    def __exit__(self, exc_type, exc, tb) -> bool:
        real = time.perf_counter() - self._t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.m.wall_real_s = real
        self.m.wall_sim_s = real * TIME_SCALE
        self.m.peak_mem_mb = peak / (1024 * 1024)
        self.m.rate_limited = sum(p.rate_limited for p in self.providers.values())
        self.m.provider_peak_in_flight = {k: p.peak_in_flight for k, p in self.providers.items()}
        if exc is not None:
            self.m.crashed = True
            self.m.crash_reason = f"{type(exc).__name__}: {exc}"
        return True  # swallow so the benchmark can continue to the next stage


def build_context(search: list[str], skus: list[dict], style: str) -> str:
    return f"{style}; {', '.join(search)}; {len(skus)} skus"


def run_stage(stage_main, stage: str, description: str, n: int = DEFAULT_N) -> Metrics:
    """Run one stage's `main(n, providers, run)` coroutine or function, return metrics."""
    providers = make_providers()
    with Run(stage, description, n, providers) as run:
        if asyncio.iscoroutinefunction(stage_main):
            asyncio.run(stage_main(n, providers, run))
        else:
            stage_main(n, providers, run)
        # A batch API only hands results back when the whole batch is done —
        # so from the caller's point of view the first result arrives at the end.
        if not run.m.crashed:
            run.mark_first_result()
    return run.m


__all__ = [
    "BRIEF", "DEFAULT_N", "Metrics", "Run", "run_stage", "build_context",
    "ImageProvider", "ImageResult", "PlannerLLM", "PlannedJob", "ToolBox", "make_providers", "TIME_SCALE", "sim", "sim_sleep",
]
