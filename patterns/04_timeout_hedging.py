"""Pattern 4 — Timeouts and hedged requests

USE CASE
    One provider request hangs for 300 s. Your batch of 40 is otherwise done
    in 20 s — and sits there for another 280 waiting on a single socket.
    For an interactive agent, p99 latency IS the user experience.

NAIVE       await provider.generate(...)   — waits as long as it takes
ASYNCIO     asyncio.timeout(30)            — hard cap on every call
            hedging                        — if primary is slow, race a backup
                                             and cancel whichever loses

Run:  python patterns/04_timeout_hedging.py
"""
import asyncio

from _setup import timed
from backends import ImageResult, make_providers, sim

providers = make_providers(hang_job=7)          # job 7 hangs on gpt_image for 300 s
gpt, gemini = providers["gpt_image"], providers["gemini"]


# --- 4a: timeout -------------------------------------------------------------
async def with_timeout(job_id: int) -> ImageResult | None:
    try:
        async with asyncio.timeout(sim(30)):                    # ← 30 simulated seconds, max
            return await gpt.generate(job_id, "look")
    except TimeoutError:
        print(f"  job {job_id}: gpt_image timed out after 30 s → mark for retry")
        return None


# --- 4b: hedging -------------------------------------------------------------
async def hedged(job_id: int, hedge_after: float = 9.0) -> ImageResult:
    """Start primary. If it hasn't answered in `hedge_after` s, start backup too.
    Return the first success; cancel the rest."""
    primary = asyncio.create_task(gpt.generate(job_id, "look"))
    tasks = {primary}
    try:
        done, _ = await asyncio.wait(tasks, timeout=sim(hedge_after))
        if not done:
            print(f"  job {job_id}: primary slow → hedging to gemini")
            tasks.add(asyncio.create_task(gemini.generate(job_id, "look")))
        while tasks:
            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if t.exception() is None:
                    return t.result()
        raise RuntimeError("all providers failed")
    finally:
        for t in tasks:
            t.cancel()                                          # ← no zombie requests


async def main() -> None:
    print("Pattern 4 — timeouts and hedging\n")
    print("A. plain await on the hanging job would take 300 s. With a timeout:")
    with timed("asyncio.timeout(30) on hanging job"):
        await with_timeout(7)

    print("\nB. hedging: primary hangs, backup answers")
    with timed("hedged(job 7)"):
        r = await hedged(7)
    print(f"  → served by {r.provider} in {r.latency_sim:.1f} s")

    print("\nC. hedging on a healthy job: primary wins, no backup is ever sent")
    gemini.reset()
    with timed("hedged(job 3)"):
        r = await hedged(3)
    print(f"  → served by {r.provider}; gemini calls made: {gemini.calls}")


if __name__ == "__main__":
    asyncio.run(main())
