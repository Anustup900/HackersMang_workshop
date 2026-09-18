"""v6 — Pattern 6: streaming with async generators.

v5 is fast and reliable, but it is still a *batch*: the caller gets a list
when everything is done. For an interactive agent that is a 40-second spinner.

Change the API shape: `run_campaign()` is now an `async def` that `yield`s
events — planner tokens as they stream, then one `JobDone` / `JobFailed`
event the instant each image clears QA. The UI, the CLI, the websocket
handler all just do `async for event in run_campaign(...)`.

Under the hood nothing about the concurrency changed; workers push into a
results queue and the generator drains it. Same pattern, same primitives.

What changes in the benchmark:
  * time-to-first-result drops from "the end" to ~first image latency
  * the caller can render progress, cancel early, or tee results to storage
"""
import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

from common import DEFAULT_N, ImageResult, PlannerLLM, Run, ToolBox, make_providers, run_stage
from v1_taskgroup import gather_context
from v3_queue import N_WORKERS, QUEUE_SIZE, _DONE, producer
from v5_retry import generate_with_retry


# ------------------------------- events ------------------------------------ #
@dataclass
class Token:
    text: str


@dataclass
class JobDone:
    result: ImageResult


@dataclass
class JobFailed:
    job_id: int
    error: str


@dataclass
class Finished:
    succeeded: int
    failed: int


Event = Token | JobDone | JobFailed | Finished


# ------------------------------- pipeline ---------------------------------- #
async def worker(q: asyncio.Queue, out: asyncio.Queue, providers, limits, tools: ToolBox, run: Run) -> None:
    while (job := await q.get()) is not _DONE:
        try:
            img = await generate_with_retry(job, providers, limits, run)
            await tools.qa_check(img)
            await out.put(JobDone(img))
        except Exception as exc:
            await out.put(JobFailed(job.job_id, f"{type(exc).__name__}: {exc}"))
        finally:
            q.task_done()
            del job


async def run_campaign(n: int, providers, run: Run) -> AsyncIterator[Event]:   # ← Pattern 6
    tools = ToolBox()
    planner = PlannerLLM(list(providers))

    async for tok in planner.stream_tokens("plan the campaign"):               # stream the LLM
        yield Token(tok)

    context = await gather_context(tools)
    limits = {name: asyncio.Semaphore(p.concurrency_limit) for name, p in providers.items()}
    q: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_SIZE)
    out: asyncio.Queue[Event] = asyncio.Queue()

    async def pipeline() -> None:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(producer(planner, n, context, q))
            for _ in range(N_WORKERS):
                tg.create_task(worker(q, out, providers, limits, tools, run))
        await out.put(None)                                                    # all workers done

    pipe = asyncio.create_task(pipeline())
    ok = bad = 0
    try:
        while (ev := await out.get()) is not None:
            if isinstance(ev, JobDone):
                ok += 1
            else:
                bad += 1
            yield ev                                                           # stream each result
        yield Finished(ok, bad)
    finally:
        pipe.cancel()                                                          # consumer walked away → tear down


# ------------------------------- consumer ---------------------------------- #
async def main(n: int, providers, run: Run) -> None:
    async for ev in run_campaign(n, providers, run):
        match ev:
            case Token(text):
                print(text, end="", flush=True)
            case JobDone(result):
                run.mark_first_result()                                        # the caller sees it NOW
                run.m.succeeded += 1
                print(f"\r  ✓ job {result.job_id:2d} via {result.provider:<15} ({run.m.succeeded}/{n})", end="", flush=True)
            case JobFailed(job_id, error):
                run.m.failed += 1
                print(f"\n  ✗ job {job_id}: {error}")
            case Finished(ok, bad):
                print(f"\n  done: {ok} ok, {bad} failed")


if __name__ == "__main__":
    print(run_stage(main, "v6", "+ streaming async generator").summary())
