"""Pattern 6 — Streaming with async generators

USE CASE
    An agent's answer is: some LLM tokens, then three tool results that
    arrive at different times, then more tokens. The UI wants to show each
    piece the moment it exists — not a spinner for 40 seconds.

NAIVE       collect everything, return a list           — user waits for the slowest part
ASYNCIO     `async def ... yield` + `async for`           — one interface for tokens,
                                                            tool results, and progress,
                                                            across every provider

Run:  python patterns/06_streaming_generators.py
"""
import asyncio
import time
from typing import AsyncIterator

from _setup import TIME_SCALE, timed
from backends import PlannerLLM, ToolBox, make_providers

tools = ToolBox()
planner = PlannerLLM(["gpt_image"])
provider = make_providers(hang_job=None)["gpt_image"]


# --- naive: batch -------------------------------------------------------------
async def answer_batch() -> list[str]:
    tokens = [t async for t in planner.stream_tokens("plan")]       # wait for all tokens
    results = await asyncio.gather(*(provider.generate(i, "look") for i in range(6)))
    return ["".join(tokens)] + [f"image {r.job_id} ready" for r in results]


# --- asyncio: stream -----------------------------------------------------------
async def answer_stream() -> AsyncIterator[str]:
    async for tok in planner.stream_tokens("plan"):                # 1. LLM tokens as they come
        yield tok

    tasks = [asyncio.create_task(provider.generate(i, "look")) for i in range(6)]
    for fut in asyncio.as_completed(tasks):                         # 2. tool results in completion order
        r = await fut
        yield f"\n  image {r.job_id} ready ({r.latency_sim:.1f} s)"

    yield "\n  all done"                                            # 3. epilogue


async def merge(*streams: AsyncIterator[str]) -> AsyncIterator[str]:
    """Bonus: fan-in several async generators into one. Any provider's stream
    can be plugged in here — that is what 'unifies streaming across providers' means."""
    q: asyncio.Queue[str | None] = asyncio.Queue()

    async def pump(s):
        async for item in s:
            await q.put(item)
        await q.put(None)

    async with asyncio.TaskGroup() as tg:
        for s in streams:
            tg.create_task(pump(s))
        live = len(streams)
        while live:
            item = await q.get()
            if item is None:
                live -= 1
            else:
                yield item


async def main() -> None:
    print("Pattern 6 — async generators for streaming\n")
    t0 = time.perf_counter()
    with timed("batch: first byte arrives at"):
        out = await answer_batch()
    print(f"  {out[0]!r} ... +{len(out) - 1} results, all at once\n")

    print("stream:")
    t0 = time.perf_counter()
    first = None
    async for chunk in answer_stream():
        if first is None:
            first = (time.perf_counter() - t0) * TIME_SCALE
        print(chunk, end="", flush=True)
    print(f"\n  first byte after {first:.1f} s simulated (vs. everything at the end)\n")

    print("merged streams from two 'providers':")
    async for chunk in merge(planner.stream_tokens("a"), planner.stream_tokens("b")):
        print(chunk, end="|", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
