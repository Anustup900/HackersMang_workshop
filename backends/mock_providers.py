from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

# One simulated second == 1 / TIME_SCALE real seconds.
# TIME_SCALE=25 means a 300 s provider hang costs you 12 real seconds.
TIME_SCALE = float(os.environ.get("TIME_SCALE", "25"))


def sim(seconds: float) -> float:
    """Convert simulated seconds to real seconds."""
    return seconds / TIME_SCALE


async def sim_sleep(seconds: float) -> None:
    await asyncio.sleep(sim(seconds))


def sim_sleep_sync(seconds: float) -> None:
    time.sleep(sim(seconds))


# --------------------------------------------------------------------------- #
# Errors — the three kinds that matter for a retry policy
# --------------------------------------------------------------------------- #
class RateLimitError(Exception):
    """HTTP 429. Retry after backing off; better: never send it."""


class TransientError(Exception):
    """5xx / connection reset. Safe to retry immediately."""


class PermanentError(Exception):
    """4xx content policy, invalid prompt. Retrying is pointless."""


# --------------------------------------------------------------------------- #
# Image providers
# --------------------------------------------------------------------------- #
@dataclass
class ImageResult:
    job_id: int
    provider: str
    latency_sim: float
    attempt: int
    payload: bytes = field(repr=False, default=b"")


@dataclass
class ImageProvider:
    """A fake image-generation API with a production-shaped failure profile."""

    name: str
    p50_latency: float          # simulated seconds
    jitter: float               # +/- fraction of p50
    concurrency_limit: int      # requests in flight before 429
    transient_rate: float       # fraction of first attempts that fail transiently
    permanent_rate: float       # fraction of jobs that can never succeed
    hang_jobs: frozenset[int]   # job ids that hang on their first attempt
    hang_seconds: float = 300.0
    seed: int = 0

    # telemetry
    in_flight: int = 0
    peak_in_flight: int = 0
    calls: int = 0
    rate_limited: int = 0

    def _rng(self, job_id: int, attempt: int) -> random.Random:
        return random.Random(hash((self.seed, job_id, attempt)))

    def _plan_call(self, job_id: int, attempt: int) -> tuple[float, Exception | None]:
        """Decide latency and outcome up front so sync and async agree."""
        rng = self._rng(job_id, attempt)
        latency = self.p50_latency * (1 + rng.uniform(-self.jitter, self.jitter))
        if job_id in self.hang_jobs and attempt == 1:
            return self.hang_seconds, None
        if rng.random() < self.permanent_rate:
            return latency * 0.2, PermanentError(f"{self.name}: prompt rejected (job {job_id})")
        if rng.random() < self.transient_rate:
            return latency * 0.5, TransientError(f"{self.name}: 503 upstream (job {job_id}, attempt {attempt})")
        return latency, None

    def _enter(self) -> None:
        self.calls += 1
        if self.in_flight >= self.concurrency_limit:
            self.rate_limited += 1
            raise RateLimitError(f"{self.name}: 429 ({self.in_flight} in flight, limit {self.concurrency_limit})")
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)

    def _exit(self) -> None:
        self.in_flight -= 1

    async def generate(self, job_id: int, prompt: str, attempt: int = 1) -> ImageResult:
        self._enter()
        try:
            latency, err = self._plan_call(job_id, attempt)
            await sim_sleep(latency)
            if err:
                raise err
            return ImageResult(job_id, self.name, latency, attempt, payload=b"\x89PNG" + bytes(64))
        finally:
            self._exit()

    def generate_sync(self, job_id: int, prompt: str, attempt: int = 1) -> ImageResult:
        self._enter()
        try:
            latency, err = self._plan_call(job_id, attempt)
            sim_sleep_sync(latency)
            if err:
                raise err
            return ImageResult(job_id, self.name, latency, attempt, payload=b"\x89PNG" + bytes(64))
        finally:
            self._exit()

    def reset(self) -> None:
        self.in_flight = self.peak_in_flight = self.calls = self.rate_limited = 0


def make_providers(hang_job: int | None = 15) -> dict[str, ImageProvider]:
    """Three providers with distinct personalities. Same every run (seeded)."""
    hang = frozenset({hang_job}) if hang_job is not None else frozenset()
    return {
        "gpt_image": ImageProvider(
            name="gpt_image", p50_latency=6.0, jitter=0.3, concurrency_limit=8,
            transient_rate=0.05, permanent_rate=0.0, hang_jobs=hang, seed=1,
        ),
        "gemini": ImageProvider(
            name="gemini", p50_latency=4.0, jitter=0.4, concurrency_limit=12,
            transient_rate=0.08, permanent_rate=0.0, hang_jobs=frozenset(), seed=2,
        ),
        "qwen_selfhosted": ImageProvider(
            name="qwen_selfhosted", p50_latency=9.0, jitter=0.2, concurrency_limit=4,
            transient_rate=0.02, permanent_rate=0.0, hang_jobs=frozenset(), seed=3,
        ),
    }


# --------------------------------------------------------------------------- #
# Agent tools — the "gather context before reasoning" step
# --------------------------------------------------------------------------- #
class ToolBox:
    """Three tools an agent typically calls before it can plan anything."""

    async def web_search(self, query: str) -> list[str]:
        await sim_sleep(0.8)
        return [f"trend: {t}" for t in ("oversized blazers", "sage green", "linen sets")]

    async def db_query(self, sql: str) -> list[dict]:
        await sim_sleep(0.3)
        return [{"sku": f"SKU-{i:04d}", "category": "blazer"} for i in range(5)]

    async def vision_describe(self, image_ref: str) -> str:
        await sim_sleep(1.5)
        return "studio, soft key light, neutral grey backdrop, 3/4 body shot"

    async def qa_check(self, result: ImageResult) -> bool:
        """A vision model looking for deformed hands, extra limbs, etc."""
        await sim_sleep(1.0)
        return True

    # sync twins for the naive version
    def web_search_sync(self, query: str) -> list[str]:
        sim_sleep_sync(0.8)
        return [f"trend: {t}" for t in ("oversized blazers", "sage green", "linen sets")]

    def db_query_sync(self, sql: str) -> list[dict]:
        sim_sleep_sync(0.3)
        return [{"sku": f"SKU-{i:04d}", "category": "blazer"} for i in range(5)]

    def vision_describe_sync(self, image_ref: str) -> str:
        sim_sleep_sync(1.5)
        return "studio, soft key light, neutral grey backdrop, 3/4 body shot"

    def qa_check_sync(self, result: ImageResult) -> bool:
        sim_sleep_sync(1.0)
        return True


# --------------------------------------------------------------------------- #
# Planner LLM — fast, produces prompts, can stream tokens
# --------------------------------------------------------------------------- #
PAYLOAD_BYTES = 2 * 1024 * 1024  # each planned job carries a 2 MB reference image


@dataclass
class PlannedJob:
    job_id: int
    prompt: str
    provider: str
    reference: bytes = field(repr=False)


class PlannerLLM:
    """Emits one planned job every 0.15 simulated seconds — much faster than
    any image backend can consume them. That gap is where backpressure lives."""

    def __init__(self, providers: list[str]):
        self.providers = providers

    def plan_one(self, i: int, context: str) -> PlannedJob:
        return PlannedJob(
            job_id=i,
            prompt=f"[{context[:20]}...] look {i}: editorial fashion shot",
            provider=self.providers[i % len(self.providers)],
            reference=bytes(PAYLOAD_BYTES),
        )

    async def plan(self, n: int, context: str) -> AsyncIterator[PlannedJob]:
        for i in range(n):
            await sim_sleep(0.15)
            yield self.plan_one(i, context)

    def plan_sync(self, n: int, context: str) -> list[PlannedJob]:
        out = []
        for i in range(n):
            sim_sleep_sync(0.15)
            out.append(self.plan_one(i, context))
        return out

    async def stream_tokens(self, prompt: str) -> AsyncIterator[str]:
        for tok in ("Planning", " a", " 40-look", " editorial", " campaign", "…"):
            await sim_sleep(0.2)
            yield tok
