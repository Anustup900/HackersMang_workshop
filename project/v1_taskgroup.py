"""v1 — go async. Pattern 1: TaskGroup for the parallel tool calls.

Two changes from v0:
  * The three context tools run concurrently inside `asyncio.TaskGroup`.
    Latency drops from 0.8 + 0.3 + 1.5 = 2.6 s to max(...) = 1.5 s, and if one
    tool raises, the others are cancelled — nothing leaks.
  * Every image job is fired at once with `gather(return_exceptions=True)`.

This is the "just add async" version — and it makes one thing WORSE:
  * all 40 jobs hit the providers at the same instant → most get 429s.
    That is the problem Pattern 2 (Semaphore) fixes.
The 300 s hang is still there, still holding the whole batch hostage.
"""
import asyncio

from common import BRIEF, ImageResult, PlannerLLM, Run, ToolBox, build_context, run_stage


async def gather_context(tools: ToolBox) -> str:
    async with asyncio.TaskGroup() as tg:                       # ← Pattern 1
        t_search = tg.create_task(tools.web_search(BRIEF))
        t_skus = tg.create_task(tools.db_query("select * from catalog"))
        t_style = tg.create_task(tools.vision_describe("reference.jpg"))
    # leaving the block means every task finished (or the group raised)
    return build_context(t_search.result(), t_skus.result(), t_style.result())


async def render_one(job, providers, tools: ToolBox, run: Run) -> ImageResult:
    img = await providers[job.provider].generate(job.job_id, job.prompt)
    await tools.qa_check(img)
    return img


async def main(n: int, providers, run: Run) -> None:
    tools = ToolBox()
    context = await gather_context(tools)

    planner = PlannerLLM(list(providers))
    jobs = [j async for j in planner.plan(n, context)]        # all 40 in memory

    outcomes = await asyncio.gather(
        *(render_one(j, providers, tools, run) for j in jobs),
        return_exceptions=True,
    )
    for o in outcomes:
        if isinstance(o, Exception):
            run.m.failed += 1
        else:
            run.m.succeeded += 1


if __name__ == "__main__":
    print(run_stage(main, "v1", "async + TaskGroup, unbounded fan-out").summary())
