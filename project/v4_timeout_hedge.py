"""v4 — Pattern 4: timeouts and hedged requests.

Job 15 hangs on gpt_image for 300 s. In v0–v3 the entire batch waits for it.

Two stdlib moves:
  1. `asyncio.timeout(30)` around every provider call — no call may take
     longer than 30 simulated seconds, full stop.
  2. Hedging: if the primary provider has not answered within
     1.5 × its p50, fire the same job at a fallback provider and take
     whichever returns first. Cancel the loser (its semaphore slot is
     released by `async with` on cancellation — that is why we use it).

What changes in the benchmark:
  * wall time collapses from ~300 s to the throughput-bound value
  * p99 latency drops — the slow tail is now "the faster of two providers"
"""
import asyncio

from common import ImageResult, PlannedJob, PlannerLLM, Run, ToolBox, run_stage, sim
from v1_taskgroup import gather_context
from v3_queue import N_WORKERS, QUEUE_SIZE, _DONE, producer

CALL_TIMEOUT = 30.0  # simulated seconds


async def call_provider(name: str, job: PlannedJob, providers, limits, attempt: int = 1) -> ImageResult:
    async with limits[name]:
        async with asyncio.timeout(sim(CALL_TIMEOUT)):                 # ← Pattern 4a
            return await providers[name].generate(job.job_id, job.prompt, attempt)


async def hedged_generate(job: PlannedJob, providers, limits, attempt: int = 1) -> ImageResult:
    """Race primary vs. a late-starting backup; first success wins."""
    order = list(providers)
    primary = job.provider
    backup = order[(order.index(primary) + 1) % len(order)]
    hedge_after = providers[primary].p50_latency * 1.5

    tasks = [asyncio.create_task(call_provider(primary, job, providers, limits, attempt), name=f"{primary}:{job.job_id}")]
    try:
        done, _ = await asyncio.wait(tasks, timeout=sim(hedge_after))
        if not done:                                                   # primary is slow → hedge
            tasks.append(asyncio.create_task(call_provider(backup, job, providers, limits, attempt), name=f"{backup}:{job.job_id}"))
        last_error: Exception | None = None
        while tasks:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                tasks.remove(t)
                if t.exception() is None:
                    return t.result()                                  # ← Pattern 4b: first success
                last_error = t.exception()
        raise last_error  # both failed
    finally:
        for t in tasks:                                                # cancel the loser(s)
            t.cancel()


async def worker(q: asyncio.Queue, providers, limits, tools: ToolBox, run: Run) -> None:
    while (job := await q.get()) is not _DONE:
        try:
            img = await hedged_generate(job, providers, limits)
            await tools.qa_check(img)
            run.m.succeeded += 1
        except Exception:
            run.m.failed += 1
        finally:
            q.task_done()
            del job


async def main(n: int, providers, run: Run) -> None:
    tools = ToolBox()
    context = await gather_context(tools)
    limits = {name: asyncio.Semaphore(p.concurrency_limit) for name, p in providers.items()}
    q: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_SIZE)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(producer(PlannerLLM(list(providers)), n, context, q))
        for _ in range(N_WORKERS):
            tg.create_task(worker(q, providers, limits, tools, run))


if __name__ == "__main__":
    print(run_stage(main, "v4", "+ timeout + hedging").summary())
