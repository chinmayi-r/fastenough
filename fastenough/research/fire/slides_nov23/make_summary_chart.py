"""Bar chart for the summary slide: mean score of every agent on rooms 0 to 49, simulator, 8 items.

It runs no agent. The means are the ones in NOTES.md (entries of Nov 23): first run, run 1, jev_run50,
opus_run50_fire, opus_run50_fire_measured.
Run:  venv_kinder\\Scripts\\python.exe results\\slides_nov23\\make_summary_chart.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
INK, MUTED, BLUE, PAGE, GRID = "#1A1D21", "#5B6470", "#1D4ED8", "#FBFBFA", "#D9DDE2"

ROWS = [  # name, mean score, is the reflex
    ("Opus before every command,\ncharged 0.15 s per call", 0.861, False),
    ("Planner,\nthinking time not charged", 0.816, False),
    ("Liane's method:\nprogram written before the run", 0.685, False),
    ("Jev before every command", 0.681, False),
    ("RRARA version 1 with Opus", 0.628, False),
    ("Reflex", 0.598, True),
    ("RRARA version 1 with Jev", 0.578, False),
    ("Opus before every command,\ncharged its real time", 0.376, False),
    ("Random", 0.041, False),
]


def main() -> None:
    plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(8.0, 6.4), dpi=200)
    fig.patch.set_facecolor(PAGE)
    ax.set_facecolor(PAGE)
    ys = list(range(len(ROWS)))[::-1]
    for y, (name, mean, is_reflex) in zip(ys, ROWS):
        ax.barh(y, mean, height=0.62, color=INK if is_reflex else BLUE, zorder=3)
        ax.text(mean + 0.015, y, f"{mean:.2f}", va="center", ha="left", fontsize=15, fontweight="bold", color=INK, zorder=5,
                bbox=dict(facecolor=PAGE, edgecolor="none", pad=1.5))
    ax.axvline(0.598, color=INK, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.set_yticks(ys, [r[0] for r in ROWS], fontsize=13, color=INK)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="x", labelsize=12, colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score: share of the value saved", fontsize=13, color=MUTED, labelpad=8)
    ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    fig.tight_layout()
    out = HERE / "summary_chart.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    print("wrote", out, fig.get_size_inches() * fig.dpi)


if __name__ == "__main__":
    main()
