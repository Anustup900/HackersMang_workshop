"""v3 — Pattern 3: backpressure with a bounded asyncio.Queue.

v1/v2 materialise every planned job in a list before rendering starts. Each
planned job carries a 2 MB reference image, so 40 jobs = 80 MB sitting idle,
4 000 jobs = 8 GB and an OOM-killed worker.

Fix: the planner PUTS into `asyncio.Queue(maxsize=4)`; a small worker pool
GETS from it. When the queue is full, `await queue.put()` blocks — the fast
producer is automatically slowed to the consumers' pace. Memory is now
proportional to (queue size + workers), not to batch size.

What changes in the benchmark:
  * peak memory falls and — crucially — stops depending on N
  * everything else stays roughly the same (the hang is still there)
"""
import asyncio

from common import PlannedJob, PlannerLLM, Run, ToolBox, run_stage
from v1_taskgroup import gather_context

QUEUE_SIZE = 4
N_WORKERS = 12
_DONE = None  # sentinel


async def producer(planner: PlannerLLM, n: int, context: str, q: asyncio.Queue) -> None:
    async for job in planner.plan(n, context):
        await q.put(job)                                        # ← blocks when full: backpressure
    for _ in range(N_WORKERS):
        await q.put(_DONE)


async def worker(q: asyncio.Queue, providers, limits, tools: ToolBox, run: Run) -> None:
    while (job := await q.get()) is not _DONE:
        try:
            async with limits[job.provider]:
                img = await providers[job.provider].generate(job.job_id, job.prompt)
            await tools.qa_check(img)
            run.m.succeeded += 1
        except Exception:
            run.m.failed += 1
        finally:
            q.task_done()
            del job                                             # drop the 2 MB reference immediately


async def main(n: int, providers, run: Run) -> None:
    tools = ToolBox()
    context = await gather_context(tools)
    limits = {name: asyncio.Semaphore(p.concurrency_limit) for name, p in providers.items()}
    q: asyncio.Queue[PlannedJob | None] = asyncio.Queue(maxsize=QUEUE_SIZE)   # ← Pattern 3

    async with asyncio.TaskGroup() as tg:
        tg.create_task(producer(PlannerLLM(list(providers)), n, context, q))
        for _ in range(N_WORKERS):
            tg.create_task(worker(q, providers, limits, tools, run))


if __name__ == "__main__":
    print(run_stage(main, "v3", "+ bounded Queue (backpressure)").summary())
