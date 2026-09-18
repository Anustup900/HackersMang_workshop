# Asyncio Is All Your Agents Need

Companion repo for the 60-minute workshop *Scaling Multi-Model AI Pipelines with
the Standard Library*. Python 3.11+, **zero dependencies** (pytest optional).

Everything runs offline against mock model backends that behave like real ones:
per-provider latency, hard concurrency caps that 429, transient 503s, and one
request that hangs for 300 seconds. Time is scaled so a full run takes seconds.

```
git clone <repo> && cd asyncio-agents-workshop
python bench/run_benchmark.py --chart      # the whole story in one table (~35 s)
python patterns/01_taskgroup_tools.py      # one pattern at a time
python project/v6_streaming.py             # the finished agent
TIME_SCALE=200 python -m pytest -q         # 10 tests, ~4 s
```

## Layout

```
backends/          fake providers: gpt_image, gemini, qwen_selfhosted + agent tools + planner LLM
patterns/          one runnable file per pattern — naive vs asyncio, side by side
  01_taskgroup_tools.py          parallel tool calls, failure isolation
  02_semaphore_ratelimit.py      rate limits as Semaphores
  03_queue_backpressure.py       bounded Queue, flat memory
  04_timeout_hedging.py          asyncio.timeout + hedged requests
  05_partial_failure_retry.py    gather(return_exceptions=True) + typed RetryPolicy
  06_streaming_generators.py     async generators, as_completed, stream fan-in
project/           the campaign agent, built up one pattern at a time
  v0_naive.py       sequential, blocking — crashes at job 2
  v1_taskgroup.py   + Pattern 1 — fast tools, but 429 storm
  v2_semaphore.py   + Pattern 2 — zero 429s
  v3_queue.py       + Pattern 3 — memory flat
  v4_timeout_hedge.py + Pattern 4 — the 300 s hang no longer matters
  v5_retry.py       + Pattern 5 — 40/40 delivered
  v6_streaming.py   + Pattern 6 — first result at 8 s, not 42 s
  exercise/         the 15-minute hands-on (see below)
bench/             run every stage on the same batch; prints table, writes results.json / benchmark.png
tests/             plain pytest, asyncio.run() — no plugins
handout/           attendee cheat-sheet
```

## The agent

> Given a brief, gather context with three tools (web search, catalog DB, vision
> model on a reference image), plan 40 looks with an LLM, render each look on one
> of three image backends, QA each output with a vision model, report progress.

## What the benchmark shows (40 jobs, TIME_SCALE=25)

| stage | change | wall (sim s) | delivered | 429s | peak mem | first result |
|---|---|---|---|---|---|---|
| v0 | sequential, blocking | crashed @ job 2 (588 projected) | 2/40 | 0 | 80 MB | never |
| v1 | + TaskGroup, unbounded gather | 309 | 22/40 | 16 | 80 MB | 309 s |
| v2 | + Semaphore per provider | 309 | 38/40 | 0 | 80 MB | 309 s |
| v3 | + bounded Queue | 309 | 38/40 | 0 | 34 MB | 309 s |
| v4 | + timeout + hedging | 37 | 38/40 | 0 | 38 MB | 37 s |
| v5 | + typed retry | 41 | 40/40 | 0 | 36 MB | 41 s |
| v6 | + streaming | 42 | 40/40 | 0 | 34 MB | **8 s** |

Every number is reproducible: the failure profile is seeded.

## Knobs

| env var | default | meaning |
|---|---|---|
| `TIME_SCALE` | 25 | simulated seconds per real second |
| `N_JOBS` | 40 | batch size for the benchmark |

Edit `backends/mock_providers.py::make_providers` to change latency, caps, failure
rates, or which job hangs.
