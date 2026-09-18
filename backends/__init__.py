"""Mock model backends with production-shaped failure modes.

Everything here is a fake: no network, no API keys. Latency is simulated with
`asyncio.sleep` (or `time.sleep` for the synchronous variants), scaled by
TIME_SCALE so a "6 second" image generation takes 6 / TIME_SCALE real seconds.

The shapes are real though — they are modelled on what a multi-model
image-generation pipeline actually sees in production:
  * different latency per provider
  * hard concurrency caps that return 429 when exceeded
  * a small percentage of transient errors (5xx, connection reset)
  * the occasional request that simply hangs
"""
from .mock_providers import (  # noqa: F401
    TIME_SCALE,
    ImageProvider,
    ImageResult,
    PermanentError,
    PlannerLLM,
    RateLimitError,
    TransientError,
    ToolBox,
    make_providers,
    sim,
    sim_sleep,
    sim_sleep_sync,
)
