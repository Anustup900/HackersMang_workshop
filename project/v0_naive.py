"""v0 — the notebook version. Everything is sequential and blocking.

This is what most agent code looks like the day it leaves the notebook:
    for each tool: call it
    for each job:  plan → generate → QA
It works on 5 images. On 40 it is slow; on 400 it is a pager alert.

What to watch in the benchmark:
  * wall time == sum of every call (no overlap at all)
  * one 300 s provider hang stalls the entire run
  * one exception at item ~37 and the whole batch is lost — no results are kept
"""
from common import BRIEF, ImageResult, PlannerLLM, Run, ToolBox, build_context, run_stage


def projected_wall_sim(n: int, providers) -> float:
    """How long v0 WOULD take if nothing crashed: the literal sum of every call."""
    total = 0.8 + 0.3 + 1.5 + n * 0.15                       # tools + planning
    names = list(providers)
    for i in range(n):
        latency, _ = providers[names[i % len(names)]]._plan_call(i, attempt=1)
        total += latency + 1.0                               # generate + QA
    return total


def main(n: int, providers, run: Run) -> None:
    run.m.notes.append(f"if it had not crashed, v0 would take {projected_wall_sim(n, providers):.0f} s (sim)")
    tools = ToolBox()

    # 1. gather context — three independent calls, run one after another
    search = tools.web_search_sync(BRIEF)
    skus = tools.db_query_sync("select * from catalog where season='AW'")
    style = tools.vision_describe_sync("reference.jpg")
    context = build_context(search, skus, style)

    # 2. plan every job up front
    planner = PlannerLLM(list(providers))
    jobs = planner.plan_sync(n, context)

    # 3. render + QA, one at a time; any exception propagates and kills the run
    results: list[ImageResult] = []
    for job in jobs:
        img = providers[job.provider].generate_sync(job.job_id, job.prompt)
        tools.qa_check_sync(img)
        results.append(img)
        run.m.succeeded += 1


if __name__ == "__main__":
    print(run_stage(main, "v0", "naive: sequential + blocking").summary())
