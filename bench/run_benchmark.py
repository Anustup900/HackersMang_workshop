"""Run every stage of the campaign agent on the same 40-job batch and print
one table. This is the "impact" slide of the workshop, generated live.

    python bench/run_benchmark.py            # table + results.json
    python bench/run_benchmark.py --chart    # also writes bench/benchmark.png
    N_JOBS=100 TIME_SCALE=50 python bench/run_benchmark.py
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))
sys.path.insert(0, str(ROOT))

from common import Metrics, run_stage  # noqa: E402

STAGES = [
    ("v0_naive", "v0", "naive: sequential + blocking"),
    ("v1_taskgroup", "v1", "async + TaskGroup, unbounded fan-out"),
    ("v2_semaphore", "v2", "+ Semaphore per provider"),
    ("v3_queue", "v3", "+ bounded Queue (backpressure)"),
    ("v4_timeout_hedge", "v4", "+ timeout + hedging"),
    ("v5_retry", "v5", "+ typed retry policy"),
    ("v6_streaming", "v6", "+ streaming async generator"),
]


def run_all(n: int) -> list[Metrics]:
    out = []
    for module, stage, desc in STAGES:
        mod = importlib.import_module(module)
        m = run_stage(mod.main, stage, desc, n)
        out.append(m)
        print(m.summary(), flush=True)
        if m.crashed:
            print(f"     ↳ {m.crash_reason}")
        for note in m.notes[:3]:
            print(f"     ↳ {note}")
    return out


def chart(metrics: list[Metrics], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, accent, grid = "#1f2933", "#7b8794", "#0f766e", "#e4e7eb"
    labels = [m.stage for m in metrics]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=200)
    fig.patch.set_facecolor("white")

    def bars(ax, values, title, unit, crashed_idx=None, fmt="{:.0f}"):
        colors = [accent] * len(values)
        if crashed_idx is not None:
            colors[crashed_idx] = muted
        b = ax.bar(labels, values, color=colors, width=0.62, zorder=3)
        ax.set_title(title, loc="left", fontsize=12, color=ink, fontweight="600", pad=10)
        ax.set_ylabel(unit, color=muted, fontsize=9)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9, length=0)
        ax.yaxis.grid(True, color=grid, zorder=0)
        for rect, v in zip(b, values):
            ax.annotate(fmt.format(v), (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha="center", va="bottom", fontsize=8.5, color=ink, xytext=(0, 2), textcoords="offset points")

    walls = [m.wall_sim_s for m in metrics]
    v0_note = next((n for n in metrics[0].notes if "would take" in n), "")
    if metrics[0].crashed and v0_note:
        walls[0] = float(v0_note.split("take ")[1].split(" ")[0])
    bars(axes[0], walls, "Wall time for 40 images", "simulated seconds", crashed_idx=0 if metrics[0].crashed else None)
    if metrics[0].crashed:
        axes[0].text(0.0, -0.16, "grey = v0 crashed at job 2; bar shows projected total", transform=axes[0].transAxes,
                     fontsize=7.5, color=muted)

    bars(axes[1], [m.succeeded for m in metrics], "Images delivered (of 40)", "")
    bars(axes[2], [m.peak_mem_mb for m in metrics], "Peak memory", "MB", fmt="{:.0f}")
    fig.tight_layout(w_pad=2.5)
    fig.savefig(path, bbox_inches="tight")
    print(f"chart → {path}")


if __name__ == "__main__":
    n = int(os.environ.get("N_JOBS", "40"))
    print(f"Campaign agent benchmark — {n} jobs, TIME_SCALE={os.environ.get('TIME_SCALE', '25')}\n")
    metrics = run_all(n)
    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps([json.loads(m.to_json()) for m in metrics], indent=2))
    print(f"\nresults → {out}")
    if "--chart" in sys.argv:
        chart(metrics, Path(__file__).with_name("benchmark.png"))
