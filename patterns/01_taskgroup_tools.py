"""Pattern 1 — Parallel tool execution with asyncio.TaskGroup

USE CASE
    An agent needs three things before it can reason: a web search, a DB
    lookup, and a vision model's description of a reference image. They are
    independent. There is no reason to wait for one before starting the next.

NAIVE       sequential awaits — latency is the SUM of the three calls
ASYNCIO     TaskGroup       — latency is the SLOWEST of the three calls,
                              and if one fails the others are cancelled
                              (structured concurrency: no orphaned tasks)

Run:  python patterns/01_taskgroup_tools.py
"""
import asyncio

from _setup import timed
from backends import ToolBox, sim_sleep

tools = ToolBox()


# --- naive ------------------------------------------------------------------
async def gather_context_sequential() -> dict:
    search = await tools.web_search("AW26 tailoring trends")       # 0.8 s
    skus = await tools.db_query("select * from catalog")           # 0.3 s
    style = await tools.vision_describe("reference.jpg")           # 1.5 s
    return {"search": search, "skus": skus, "style": style}        # total 2.6 s


# --- asyncio ----------------------------------------------------------------
async def gather_context_parallel() -> dict:
    async with asyncio.TaskGroup() as tg:
        t_search = tg.create_task(tools.web_search("AW26 tailoring trends"))
        t_skus = tg.create_task(tools.db_query("select * from catalog"))
        t_style = tg.create_task(tools.vision_describe("reference.jpg"))
    # The `async with` block only exits when ALL tasks are done.
    # If any task raised, the others are cancelled and an ExceptionGroup is raised here.
    return {"search": t_search.result(), "skus": t_skus.result(), "style": t_style.result()}


# --- failure isolation demo -------------------------------------------------
async def flaky_tool() -> None:
    await sim_sleep(0.2)
    raise RuntimeError("vision model returned 500")


async def slow_tool() -> str:
    try:
        await sim_sleep(10)                # would take 10 s ...
        return "never reached"
    except asyncio.CancelledError:
        print("  slow_tool: cancelled by the TaskGroup — not left running in the background")
        raise


async def failure_isolation() -> None:
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(flaky_tool())
            tg.create_task(slow_tool())
    except* RuntimeError as eg:           # Python 3.11 except* syntax
        print(f"  caught {len(eg.exceptions)} error(s): {eg.exceptions[0]}")


async def main() -> None:
    print("Pattern 1 — TaskGroup for parallel tool calls\n")
    with timed("sequential (sum of latencies)"):
        await gather_context_sequential()
    with timed("TaskGroup (max of latencies)"):
        await gather_context_parallel()
    print("\nFailure isolation:")
    with timed("one tool fails → siblings cancelled"):
        await failure_isolation()


if __name__ == "__main__":
    asyncio.run(main())
