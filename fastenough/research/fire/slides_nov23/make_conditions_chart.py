"""Bar chart for the three-conditions slide: random, reflex, planner with thinking free.

Reads the per-room scores the rule check already wrote. It runs no agent.
  reflex, planner : fastenough/research/fire/results/firelab_gate/agents_seeds.csv
  random          : fastenough/research/fire/results/firelab_sim/first_run.csv, column "rule" (the rule check's score, same 50 rooms)
Run:  venv_kinder\\Scripts\\python.exe results\\slides_nov23\\make_conditions_chart.py
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
INK, MUTED, BLUE, PAGE, GRID = "#1A1D21", "#5B6470", "#1D4ED8", "#FBFBFA", "#D9DDE2"


def column(path: Path, name: str, **where: str) -> np.ndarray:
    rows = [r for r in csv.DictReader(open(path)) if all(r[k] == v for k, v in where.items())]
    return np.array([float(r[name]) for r in rows])


def main() -> None:
    reflex = column(GATE / "agents_seeds.csv", "reflex", items=ITEMS, values=VALUES)
    planner = column(GATE / "agents_seeds.csv", "always_0", items=ITEMS, values=VALUES)
    random = column(HERE.parent / "results" / "firelab_sim" / "first_run.csv", "rule", agent="random")

    plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(7.6, 6.0), dpi=200)
    fig.patch.set_facecolor(PAGE)
    ax.set_facecolor(PAGE)
    names = ["Random", "Reflex", "Planner,\nthinking free"]
    data = [random, reflex, planner]
    for x, d in enumerate(data):
        mean, half = d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        ax.bar(x, mean, width=0.5, color=BLUE, zorder=3)
        ax.errorbar(x, mean, yerr=half, color=INK, linewidth=1.4, capsize=5, zorder=4)
        ax.text(x, mean + half + 0.03, f"{mean:.2f}", ha="center", va="bottom", fontsize=17, fontweight="bold", color=INK)
    ax.set_xticks(range(3), names, fontsize=15, color=INK)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=13, colors=MUTED, length=0)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_ylim(0, 1.12)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("Score: share of the value saved", fontsize=14, color=MUTED, labelpad=10)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    fig.tight_layout()
    out = HERE / "conditions_chart.png"
    fig.savefig(out, facecolor=fig.get_facecolor())

    diff = planner - reflex
    rng = np.random.default_rng(0)
    boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(4000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"rooms: random {len(random)}, reflex {len(reflex)}, planner {len(planner)}")
    print(f"means: random {random.mean():.2f}, reflex {reflex.mean():.2f}, planner {planner.mean():.2f}")
    print(f"gap planner - reflex: {diff.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]; planner better/equal/worse in {(diff > 1e-9).sum()}/{(abs(diff) <= 1e-9).sum()}/{(diff < -1e-9).sum()} rooms")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
