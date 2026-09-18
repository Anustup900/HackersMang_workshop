"""Pattern 5 — Partial failure without pipeline failure

USE CASE
    A 500-image job. 12 fail with transient 503s. With a plain loop (or a
    plain gather) the first exception kills the run — at item 37, with 36
    finished images you now have to track down and 463 you never started.

NAIVE       gather(...)                       — first exception propagates,
                                                results of the others are lost
ASYNCIO     gather(return_exceptions=True)    — every outcome comes back,
                                                exceptions as values
          + a TYPED retry policy              — retry 503s, never retry 4xx

Run:  python patterns/05_partial_failure_retry.py
"""
import asyncio
from dataclasses import dataclass

from _setup import timed
from backends import PermanentError, RateLimitError, TransientError, make_providers, sim_sleep

N_JOBS = 40
provider = make_providers(hang_job=None)["gemini"]      # 8 % transient failure rate
limit = asyncio.Semaphore(provider.concurrency_limit)


async def generate(job_id: int, attempt: int = 1):
    async with limit:
        return await provider.generate(job_id, "look", attempt)


# --- naive -------------------------------------------------------------------
async def all_or_nothing() -> None:
    try:
        results = await asyncio.gather(*(generate(i) for i in range(N_JOBS)))
        print(f"  {len(results)} results")
    except TransientError as e:
        print(f"  run killed by first error: {e}")
        print("  (the other 39 tasks kept running — but their results are gone)")


# --- asyncio: keep every outcome ---------------------------------------------
async def keep_partial() -> None:
    outcomes = await asyncio.gather(*(generate(i) for i in range(N_JOBS)), return_exceptions=True)
    ok = [o for o in outcomes if not isinstance(o, Exception)]
    bad = [(i, o) for i, o in enumerate(outcomes) if isinstance(o, Exception)]
    print(f"  {len(ok)} succeeded on first pass, {len(bad)} failed: jobs {[i for i, _ in bad]}")


# --- asyncio: typed retry policy ---------------------------------------------
@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_backoff: float = 0.5   # simulated seconds

    def should_retry(self, exc: BaseException, attempt: int) -> bool:
        if attempt >= self.max_attempts:
            return False
        if isinstance(exc, PermanentError):
            return False                                     # ← 4xx: retrying is pointless
        return isinstance(exc, (TransientError, RateLimitError, TimeoutError))

    def backoff(self, exc: BaseException, attempt: int) -> float:
        mult = 4 if isinstance(exc, RateLimitError) else 1
        return self.base_backoff * 2 ** (attempt - 1) * mult


async def with_retry(job_id: int, policy: RetryPolicy):
    attempt = 0
    while True:
        attempt += 1
        try:
            return await generate(job_id, attempt)
        except Exception as exc:
            if not policy.should_retry(exc, attempt):
                raise
            await sim_sleep(policy.backoff(exc, attempt))


async def retry_then_report() -> None:
    policy = RetryPolicy()
    outcomes = await asyncio.gather(*(with_retry(i, policy) for i in range(N_JOBS)), return_exceptions=True)
    ok = [o for o in outcomes if not isinstance(o, Exception)]
    retried = [o.job_id for o in ok if o.attempt > 1]
    bad = [(i, o) for i, o in enumerate(outcomes) if isinstance(o, Exception)]
    print(f"  {len(ok)}/{N_JOBS} succeeded; {len(retried)} needed a retry: jobs {retried}; "
          f"{len(bad)} gave up")


async def main() -> None:
    print("Pattern 5 — partial failure, typed retries\n")
    with timed("gather() — all or nothing"):
        await all_or_nothing()
    provider.reset()
    with timed("gather(return_exceptions=True)"):
        await keep_partial()
    provider.reset()
    with timed("+ RetryPolicy"):
        await retry_then_report()


if __name__ == "__main__":
    asyncio.run(main())
