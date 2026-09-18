"""v2 — Pattern 2: rate limits as Semaphores, not sleep()s.

Each provider publishes a concurrency cap (8 / 12 / 4). We mirror each cap
with one `asyncio.Semaphore` and acquire it around every call. Three lines.

What changes in the benchmark:
  * 429s drop to zero — we never send a request the provider will reject
  * peak in-flight per provider == its cap exactly (full quota utilisation)
  * wall time is now bounded by throughput, not by luck
Still to fix: the hang (Pattern 4), memory (Pattern 3), transient errors (Pattern 5).
"""
import asyncio

from common import ImageResult, PlannerLLM, Run, ToolBox, run_stage
from v1_taskgroup import gather_context


async def render_one(job, providers, limits, tools: ToolBox, run: Run) -> ImageResult:
    async with limits[job.provider]:                            # ← Pattern 2
        img = await providers[job.provider].generate(job.job_id, job.prompt)
    await tools.qa_check(img)                                   # QA is outside the limit — different resource
    return img


async def main(n: int, providers, run: Run) -> None:
    tools = ToolBox()
    context = await gather_context(tools)
    limits = {name: asyncio.Semaphore(p.concurrency_limit) for name, p in providers.items()}

    planner = PlannerLLM(list(providers))
    jobs = [j async for j in planner.plan(n, context)]

    outcomes = await asyncio.gather(
        *(render_one(j, providers, limits, tools, run) for j in jobs),
        return_exceptions=True,
    )
    for o in outcomes:
        if isinstance(o, Exception):
            run.m.failed += 1
        else:
            run.m.succeeded += 1


if __name__ == "__main__":
    print(run_stage(main, "v2", "+ Semaphore per provider").summary())
