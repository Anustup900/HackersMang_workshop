"""Pattern 2 — Rate limits as Semaphores, not sleep()s

USE CASE
    A provider allows 8 concurrent requests. Your agent has 200 images to
    render. Send them all and you get 429s; send them one at a time and you
    use 1/8 of the quota you are paying for.

NAIVE       fire everything → most requests are rejected with 429
            (or: time.sleep(0.5) between requests → guesswork, under-utilised)
ASYNCIO     Semaphore(8) around the call → exactly 8 in flight, zero 429s,
            three lines of code, nothing to tune

Run:  python patterns/02_semaphore_ratelimit.py
"""
import asyncio

from _setup import timed
from backends import RateLimitError, make_providers

N_JOBS = 40
provider = make_providers(hang_job=None)["gpt_image"]     # limit = 8, p50 = 6 s


async def unbounded() -> None:
    async def one(i: int):
        try:
            await provider.generate(i, "look")
            return True
        except RateLimitError:
            return False
    ok = await asyncio.gather(*(one(i) for i in range(N_JOBS)))
    print(f"  fire-and-hope: {sum(ok)}/{N_JOBS} succeeded, {provider.rate_limited} × 429, "
          f"peak in flight {provider.peak_in_flight}")


async def with_semaphore() -> None:
    limit = asyncio.Semaphore(provider.concurrency_limit)       # ← the whole pattern

    async def one(i: int):
        async with limit:                                       # ← acquire / release
            await provider.generate(i, "look")
            return True

    ok = await asyncio.gather(*(one(i) for i in range(N_JOBS)), return_exceptions=True)
    ok = [o for o in ok if o is True]
    print(f"  Semaphore({provider.concurrency_limit}): {len(ok)}/{N_JOBS} succeeded, {provider.rate_limited} × 429, "
          f"peak in flight {provider.peak_in_flight}")


async def main() -> None:
    print("Pattern 2 — Semaphore as a rate limiter\n")
    with timed("unbounded gather"):
        await unbounded()
    provider.reset()
    with timed("Semaphore(limit)"):
        await with_semaphore()
    print("\nNote: a few transient 503s remain — that is Pattern 5's job, not the Semaphore's.")


if __name__ == "__main__":
    asyncio.run(main())
