"""Line chart for the thinking-time slide: score against seconds charged per plan, three ways of using the planner.

Reads the per-room scores the rule check already wrote (fastenough/research/fire/results/firelab_gate/agents_seeds.csv). It runs no agent.
Run:  venv_kinder\\Scripts\\python.exe results\\slides_nov23\\make_clock_chart.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
GATE = HERE.parent / "results" / "firelab_gate"
ITEMS, VALUES = "8", "(1, 2, 3, 5, 8)"
COSTS = (0, 2, 5, 10, 20, 100)  # ticks per plan; 10 ticks = 1 s
INK, MUTED, PAGE, GRID = "#1A1D21", "#5B6470", "#FBFBFA", "#D9DDE2"
SERIES = (  # column prefix, legend text, colour, marker
    ("always", "Think before every action", "#2a78d6", "o"),
    ("commit", "Plan once, then follow the plan", "#eb6834", "s"),
    ("moving", "Reflex acts while the plan is made", "#1baf7a", "^"),
)


def main() -> None:
    rows = [r for r in csv.DictReader(open(GATE / "agents_seeds.csv")) if r["items"] == ITEMS and r["values"] == VALUES]
    mean = lambda name: float(np.mean([float(r[name]) for r in rows]))
    reflex = mean("reflex")

    plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(8.6, 6.0), dpi=200)
    fig.patch.set_facecolor(PAGE)
    ax.set_facecolor(PAGE)
    x = range(len(COSTS))
    ax.axhline(reflex, color=INK, linewidth=1.6, linestyle=(0, (5, 4)), zorder=2)
    ax.text(-0.2, reflex + 0.02, f"Reflex {reflex:.2f}", ha="left", va="bottom", fontsize=14, color=INK)
    for prefix, label, colour, marker in SERIES:
        y = [mean(f"{prefix}_{c}") for c in COSTS]
        ax.plot(x, y, color=colour, linewidth=2.4, marker=marker, markersize=9, markeredgecolor=PAGE, markeredgewidth=1.5, label=label, zorder=3)
        print(prefix, [round(v, 2) for v in y])
    ax.set_xticks(list(x), ["0", "0.2", "0.5", "1", "2", "10"], fontsize=14, color=INK)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=13, colors=MUTED, length=0)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.25, len(COSTS) - 1 + 0.25)
    ax.set_xlabel("Seconds charged per plan", fontsize=14, color=MUTED, labelpad=10)
    ax.set_ylabel("Score: share of the value saved", fontsize=14, color=MUTED, labelpad=10)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.legend(loc="lower left", frameon=False, fontsize=14, labelcolor=INK, handlelength=2.2, borderaxespad=0.6)
    fig.tight_layout()
    out = HERE / "clock_chart.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"reflex {reflex:.2f}, rooms {len(rows)}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
