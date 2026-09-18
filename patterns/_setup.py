"""Tiny shared helper so each pattern file stays focused on ONE idea."""
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backends import TIME_SCALE  # noqa: E402,F401


@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    yield
    real = time.perf_counter() - t0
    print(f"  {label:<40} {real * TIME_SCALE:7.1f} s simulated  ({real:.2f} s real)")
