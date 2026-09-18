"""Pattern 3 — Backpressure with a bounded asyncio.Queue

USE CASE
    A planner LLM emits a new image prompt (with a 2 MB reference image
    attached) every 0.15 s. The image backend takes ~6 s per item with 8
    workers. The producer is ~5x faster than the consumers. With unbounded
    fan-out, everything the planner produces sits in memory until rendered.

NAIVE       list of all planned jobs → memory grows with batch size (OOM)
ASYNCIO     Queue(maxsize=4) → put() blocks when full, producer slows to the
            consumers' pace, memory is flat regardless of batch size

Run:  python patterns/03_queue_backpressure.py
"""
import asyncio
import tracemalloc

from _setup import timed
from backends import PlannerLLM, make_providers
from backends.mock_providers import PlannedJob

N_JOBS = 60
N_WORKERS = 8
provider = make_providers(hang_job=None)["gemini"]
planner = PlannerLLM(["gemini"])


def peak_mb() -> float:
    _, peak = tracemalloc.get_traced_memory()
    return peak / 1024 / 1024


async def render(job: PlannedJob) -> None:
    try:
        await provider.generate(job.job_id, job.prompt)
    except Exception:
        pass


# --- naive: materialise everything, then fan out ---------------------------
async def unbounded() -> None:
    jobs = [j async for j in planner.plan(N_JOBS, "ctx")]          # 60 × 2 MB in memory
    limit = asyncio.Semaphore(N_WORKERS)

    async def one(j):
        async with limit:
            await render(j)
    await asyncio.gather(*(one(j) for j in jobs))


# --- asyncio: bounded queue + worker pool -----------------------------------
async def bounded() -> None:
    q: asyncio.Queue[PlannedJob | None] = asyncio.Queue(maxsize=4)      # ← the whole pattern

    async def producer():
        async for job in planner.plan(N_JOBS, "ctx"):
            await q.put(job)                                           # blocks when 4 are waiting
        for _ in range(N_WORKERS):
            await q.put(None)

    async def worker():
        while (job := await q.get()) is not None:
            await render(job)
            del job                                                    # free the 2 MB now

    async with asyncio.TaskGroup() as tg:
        tg.create_task(producer())
        for _ in range(N_WORKERS):
            tg.create_task(worker())


async def main() -> None:
    print("Pattern 3 — bounded Queue for backpressure\n")
    for label, fn in (("unbounded list + gather", unbounded), ("Queue(maxsize=4) + 8 workers", bounded)):
        provider.reset()
        tracemalloc.start()
        with timed(label):
            await fn()
        print(f"  {'':40} peak memory {peak_mb():6.1f} MB\n")
        tracemalloc.stop()
    print("Try N_JOBS = 600: the unbounded peak grows 10x; the bounded one does not move.")


if __name__ == "__main__":
    asyncio.run(main())
