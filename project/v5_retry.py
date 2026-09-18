"""v5 — Pattern 5: partial failure without pipeline failure.

Providers fail transiently 2–8 % of the time. In v3/v4 those jobs are simply
counted as failed. The fix is a *typed* retry policy: decide what to do based
on the exception class, not on a blanket `except Exception`.

    TransientError  → retry, exponential backoff, up to 3 attempts
    RateLimitError  → retry, longer backoff (we should never see it — Semaphore)
    PermanentError  → give up immediately, record, move on
    anything else   → give up, record, move on

The batch never dies: 488 succeed on the first pass, 12 retry, the report
tells you exactly which ones and why.

What changes in the benchmark:
  * success rate → 100 % (or as close as the failure profile allows)
  * wall time barely moves — retries overlap with everything else
"""
import asyncio
from dataclasses import dataclass

from common import ImageResult, PlannedJob, PlannerLLM, Run, ToolBox, run_stage, sim_sleep
from backends import PermanentError, RateLimitError, TransientError
from v1_taskgroup import gather_context
from v3_queue import N_WORKERS, QUEUE_SIZE, _DONE, producer
from v4_timeout_hedge import hedged_generate


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_backoff: float = 0.5  # simulated seconds

    def should_retry(self, exc: BaseException, attempt: int) -> bool:
        if attempt >= self.max_attempts:
            return False
        return isinstance(exc, (TransientError, RateLimitError, TimeoutError))

    def backoff(self, exc: BaseException, attempt: int) -> float:
        mult = 4 if isinstance(exc, RateLimitError) else 1
        return self.base_backoff * (2 ** (attempt - 1)) * mult


POLICY = RetryPolicy()


async def generate_with_retry(job: PlannedJob, providers, limits, run: Run) -> ImageResult:
    """Wrap the hedged call from v4 in a typed retry loop."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return await hedged_generate(job, providers, limits, attempt=attempt)
        except PermanentError:
            raise                                                      # ← never retry these
        except Exception as exc:
            if not POLICY.should_retry(exc, attempt):
                raise
            run.m.notes.append(f"job {job.job_id}: retry {attempt} after {type(exc).__name__}")
            await sim_sleep(POLICY.backoff(exc, attempt))              # ← Pattern 5


async def worker(q: asyncio.Queue, providers, limits, tools: ToolBox, run: Run) -> None:
    while (job := await q.get()) is not _DONE:
        try:
            img = await generate_with_retry(job, providers, limits, run)
            await tools.qa_check(img)
            run.m.succeeded += 1
        except Exception as exc:
            run.m.failed += 1
            run.m.notes.append(f"job {job.job_id}: FAILED permanently: {exc}")
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
    m = run_stage(main, "v5", "+ typed retry policy")
    print(m.summary())
    for note in m.notes[:10]:
        print("   ", note)
