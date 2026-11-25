"""Floor plan of one room of the fire task, for the Nov 23 slides.

Reads the room from the rule check (fastenough.tasks.fire.gate) and draws it. It runs no agent and changes nothing.
Run:  venv_kinder\\Scripts\\python.exe results\\slides_nov23\\make_plan_picture.py --seed 0 --tick 100
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from fastenough.tasks.fire.gate import NX, NY, SAFE_ROW, START, Task, make_world

CELL = 0.2
X0, Y0 = -1.5, -1.5
INK, MUTED, BLUE = "#1A1D21", "#5B6470", "#1D4ED8"
FLOOR, SOLID, FIRE, SAFE = "#ECEDEF", "#8A93A3", "#F28C28", "#CFE8D5"


def corner(i: int, j: int) -> tuple[float, float]:
    return X0 + CELL * i, Y0 + CELL * j


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--tick", type=int, default=150)
    ap.add_argument("--items", type=int, default=8)
    ap.add_argument("--out", default=str(Path(__file__).with_name("plan_room.png")))
    args = ap.parse_args()

    task = Task(items=args.items, values=(1, 2, 3, 5, 8), layout="one_wide", clearance=True, reach=1, bag=True, hp_loss=2.0)
    w = make_world(task, args.seed)

    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=200)
    fig.patch.set_facecolor("#FBFBFA")
    ax.set_facecolor("#FBFBFA")
    for i in range(NX):
        for j in range(NY):
            if (i, j) not in w.burn:
                colour = SOLID  # wall or furniture
            elif w.burn[(i, j)] <= args.tick:
                colour = FIRE
            elif j >= SAFE_ROW:
                colour = SAFE
            else:
                colour = FLOOR
            ax.add_patch(Rectangle(corner(i, j), CELL, CELL, facecolor=colour, edgecolor="#FBFBFA", linewidth=0.6))

    for (i, j), v in zip(w.cells, w.values):
        x, y = corner(i, j)
        ax.add_patch(plt.Circle((x + CELL / 2, y + CELL / 2), 0.085, facecolor="white", edgecolor=INK, linewidth=1.4, zorder=3))
        ax.text(x + CELL / 2, y + CELL / 2 - 0.005, f"{int(v)}", ha="center", va="center", fontsize=9, color=INK, zorder=4, fontweight="bold")

    sx, sy = corner(*START)
    cx, cy = sx + CELL / 2, sy + CELL / 2
    ax.add_patch(Rectangle((cx - 0.255, cy - 0.275), 0.51, 0.55, facecolor="none", edgecolor=BLUE, linewidth=2.2, zorder=5))

    ax.plot([-1.5, -0.5], [-1.62, -1.62], color=INK, linewidth=2)
    ax.text(-1.0, -1.72, "1 m", ha="center", va="top", fontsize=10, color=INK)

    ax.set_xlim(-1.55, 1.75)
    ax.set_ylim(-1.85, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(args.out, facecolor=fig.get_facecolor())
    burning = sum(1 for b in w.burn.values() if b <= args.tick)
    print(f"seed {args.seed} tick {args.tick}: items at {w.cells} worth {w.values}; {burning} of {len(w.burn)} cells burning; episode ends at tick {w.t_end}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
