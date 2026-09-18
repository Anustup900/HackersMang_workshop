"""pytest-able coroutines — no plugins needed, just asyncio.run().

    TIME_SCALE=200 python -m pytest -q
"""
import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("TIME_SCALE", "200")  # tests should be fast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "project"))

from backends import PermanentError, RateLimitError, TransientError, make_providers  # noqa: E402
from common import run_stage  # noqa: E402
import v0_naive, v2_semaphore, v3_queue, v4_timeout_hedge, v5_retry, v6_streaming  # noqa: E402,E401


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- Pattern 1
def test_taskgroup_cancels_siblings_on_failure():
    cancelled = []

    async def slow():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    async def boom():
        raise ValueError("x")

    async def main():
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(slow())
                tg.create_task(boom())
        except* ValueError:
            pass

    run(main())
    assert cancelled == [True]


# ---------------------------------------------------------------- Pattern 2
def test_semaphore_never_exceeds_provider_limit():
    p = make_providers(hang_job=None)["gpt_image"]
    sem = asyncio.Semaphore(p.concurrency_limit)

    async def one(i):
        async with sem:
            try:
                await p.generate(i, "x")
            except TransientError:
                pass

    async def main():
        await asyncio.gather(*(one(i) for i in range(50)))

    run(main())
    assert p.rate_limited == 0
    assert p.peak_in_flight == p.concurrency_limit


def test_unbounded_fanout_gets_rate_limited():
    p = make_providers(hang_job=None)["gpt_image"]

    async def one(i):
        try:
            await p.generate(i, "x")
        except (RateLimitError, TransientError):
            pass

    async def main():
        await asyncio.gather(*(one(i) for i in range(50)))

    run(main())
    assert p.rate_limited > 0


# ---------------------------------------------------------------- Pattern 3
def test_bounded_queue_uses_less_memory_than_unbounded():
    m2 = run_stage(v2_semaphore.main, "v2", "", n=30)
    m3 = run_stage(v3_queue.main, "v3", "", n=30)
    assert m3.peak_mem_mb < m2.peak_mem_mb * 0.6


# ---------------------------------------------------------------- Pattern 4
def test_hedging_beats_hanging_primary():
    providers = make_providers(hang_job=7)
    limits = {k: asyncio.Semaphore(v.concurrency_limit) for k, v in providers.items()}
    job = v3_queue.PlannerLLM(list(providers)).plan_one(7, "ctx")  # job 7 → gpt_image (hangs)
    hang_wall = providers["gpt_image"].hang_seconds
    res = run(v4_timeout_hedge.hedged_generate(job, providers, limits))
    assert res.provider == "gemini"
    assert res.latency_sim < hang_wall
    # loser was cancelled → its slot is released
    assert providers["gpt_image"].in_flight == 0


def test_timeout_raises_timeouterror():
    providers = make_providers(hang_job=7)
    limits = {k: asyncio.Semaphore(v.concurrency_limit) for k, v in providers.items()}
    job = v3_queue.PlannerLLM(list(providers)).plan_one(7, "ctx")
    try:
        run(v4_timeout_hedge.call_provider("gpt_image", job, providers, limits))
        assert False, "expected TimeoutError"
    except TimeoutError:
        pass


# ---------------------------------------------------------------- Pattern 5
def test_retry_policy_is_typed():
    pol = v5_retry.RetryPolicy(max_attempts=3)
    assert pol.should_retry(TransientError(), 1)
    assert pol.should_retry(RateLimitError(), 2)
    assert not pol.should_retry(TransientError(), 3)          # exhausted
    assert not pol.should_retry(PermanentError(), 1)          # never
    assert not pol.should_retry(ValueError(), 1)              # unknown → don't
    assert pol.backoff(RateLimitError(), 1) > pol.backoff(TransientError(), 1)


def test_v5_delivers_every_job():
    m = run_stage(v5_retry.main, "v5", "", n=40)
    assert m.succeeded == 40 and m.failed == 0 and not m.crashed


def test_v0_crashes_on_first_transient_error():
    m = run_stage(v0_naive.main, "v0", "", n=40)
    assert m.crashed and "TransientError" in m.crash_reason


# ---------------------------------------------------------------- Pattern 6
def test_streaming_yields_results_before_the_end():
    providers = make_providers(hang_job=None)

    async def main():
        from common import Run
        with Run("v6", "", 12, providers) as r:
            seen = []
            async for ev in v6_streaming.run_campaign(12, providers, r):
                seen.append(type(ev).__name__)
        return seen

    seen = run(main())
    assert seen[0] == "Token"
    assert seen[-1] == "Finished"
    assert seen.count("JobDone") == 12
