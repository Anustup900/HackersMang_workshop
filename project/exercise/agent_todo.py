"""HANDS-ON (15 min) — make this agent production-grade with 3 small edits.

This is the v1 agent: async, but naive. Run it first:

    python project/exercise/agent_todo.py

You will see: ~22/40 images, 16 × 429, and a 309 s wall clock because one
request hangs. Fix it, one TODO at a time, re-running after each:

  TODO 1  (3 lines)  Semaphore per provider          → 429s go to zero
  TODO 2  (2 lines)  asyncio.timeout around the call → wall time collapses
  TODO 3  (~6 lines) retry TransientError/TimeoutError once → 40/40 delivered

Target after all three:  ok=40/40  429s=0  wall < 60 s (sim).
Stuck? `agent_solution.py` sits next to this file.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backends import PermanentError, TransientError  # noqa: E402,F401
from common import BRIEF, ImageResult, PlannerLLM, Run, ToolBox, build_context, run_stage, sim  # noqa: E402

CALL_TIMEOUT = 30.0  # simulated seconds


async def gather_context(tools: ToolBox) -> str:
    async with asyncio.TaskGroup() as tg:
        t_search = tg.create_task(tools.web_search(BRIEF))
        t_skus = tg.create_task(tools.db_query("select * from catalog"))
        t_style = tg.create_task(tools.vision_describe("reference.jpg"))
    return build_context(t_search.result(), t_skus.result(), t_style.result())


async def render_one(job, providers, limits, tools: ToolBox) -> ImageResult:
    provider = providers[job.provider]

    # TODO 1: acquire the provider's semaphore around the generate() call
    #         hint:  async with limits[job.provider]:
    # TODO 2: cap the call at CALL_TIMEOUT simulated seconds
    #         hint:  async with asyncio.timeout(sim(CALL_TIMEOUT)):
    # TODO 3: on TransientError or TimeoutError, try once more (attempt=2)
    #         hint:  a small for-loop over attempt in (1, 2) with try/except
    img = await provider.generate(job.job_id, job.prompt)

    await tools.qa_check(img)
    return img


async def main(n: int, providers, run: Run) -> None:
    tools = ToolBox()
    context = await gather_context(tools)

    # TODO 1 (part a): build one Semaphore per provider, sized to its concurrency_limit
    limits = {}

    planner = PlannerLLM(list(providers))
    jobs = [j async for j in planner.plan(n, context)]

    outcomes = await asyncio.gather(
        *(render_one(j, providers, limits, tools) for j in jobs),
        return_exceptions=True,
    )
    for o in outcomes:
        if isinstance(o, Exception):
            run.m.failed += 1
            run.m.notes.append(f"{type(o).__name__}: {o}")
        else:
            run.m.succeeded += 1


if __name__ == "__main__":
    m = run_stage(main, "todo", "hands-on agent")
    print(m.summary())
    if m.failed:
        print(f"  first failure: {m.notes[0]}")
    target_ok = m.succeeded == m.n_jobs and m.rate_limited == 0 and m.wall_sim_s < 60
    print("\n  ✅ production-grade!" if target_ok else "\n  ⏳ not there yet — pick the next TODO")
