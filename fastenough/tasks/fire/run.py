"""The plain runner for the fire task. PROVISIONAL (Nov 26; PLAN.md section 10).

Plays random, the reflex, the reference planner or a program in the simulator, and writes one row per room and
agent into a folder you name. It imports nothing from fastenough/research, so it stays when that folder is removed.

PROVISIONAL means: built from the fire task alone, before a second task exists. Its settings and its file format
may change. The stored results all come from the research runner (fastenough/research/fire/run_rooms.py), not
from this file. `planner` here is that runner's agent `always` with the clock off: thinking costs nothing.

Run: python -m fastenough.tasks.fire.run --agents random reflex planner --first 0 --rooms 5 --out my_run
     python -m fastenough.tasks.fire.run --program my_policy.py --first 50 --rooms 5 --out my_run
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import fastenough.tasks.fire  # noqa: F401
from fastenough.tasks.fire.agents import random_agent, reflex
from fastenough.tasks.fire.clock import Clock
from fastenough.tasks.fire.env import FireLab3DEnv
from fastenough.tasks.fire.gate import paired_interval
from fastenough.tasks.fire.interface import Session
from fastenough.tasks.fire.programs import load_program, run_program
from fastenough.tasks.fire.skills import Skills
from fastenough.tasks.fire.thinkers import thinker

AGENTS = ("random", "reflex", "planner")
COLUMNS = ["room", "agent", "score", "ticks", "health", "died", "commands", "failed", "illegal", "raised", "decide_s"]


def play_room(env, skills: Skills, room: int, agent: str, module=None) -> dict:
    """One room, one agent, one row. `agent` is one of AGENTS, or "program" with the loaded `module`."""
    env.reset(seed=room)
    skills.start()
    illegal, raised, decide_s = 0, 0, ""  # decide_s stays empty where the time to decide is not measured
    if agent == "program":
        log = run_program(Session(skills), module)
        illegal = sum(r["reason"].startswith("illegal") for r in log)
        raised = sum(bool(r["error"]) for r in log)
        failed = sum((not r["done"]) and not r["reason"].startswith("illegal") for r in log)
        decide_s = round(sum(r["seconds"] for r in log), 5)
    else:
        if agent == "random":
            log = random_agent(skills, 10_000 + room)  # the seed the research runner and the rule check use
        elif agent == "reflex":
            log = reflex(skills)
        else:
            thoughts: list = []
            log = thinker(skills, Clock(), thoughts=thoughts)  # plan before every command, clock off
            decide_s = round(sum(t["measured_s"] for t in thoughts), 4)
        failed = sum(not result.done for _, _, _, result in log)
    health = skills.info.get("health", 100.0)
    return {"room": room, "agent": agent, "score": round(skills.score(), 4), "ticks": skills.ticks, "health": health,
            "died": health <= 0, "commands": len(log), "failed": failed, "illegal": illegal, "raised": raised, "decide_s": decide_s}


def play(agents, rooms, items: int = 8, values: tuple = (1, 2, 3, 5, 8), program=None) -> list:
    """Every room with every agent, and with the program if one is given. The room is built once."""
    module = load_program(program) if program else None  # raises if the audit fails: an unfit program never plays
    names = [*agents, *(["program"] if program else [])]
    env = FireLab3DEnv(level=2, num_items=items, values=values, realistic_bg=False)
    skills, rows = Skills(env), []
    try:
        for room in rooms:
            rows += [play_room(env, skills, room, name, module) for name in names]
            print(f"room {room}: " + ", ".join(f"{r['agent']} {r['score']:.3f}" for r in rows[-len(names):]), flush=True)
    finally:
        env.close()
    return rows


def summary(rows: list) -> dict:
    """Per agent: rooms, mean score, deaths; and its score minus the reflex's, room by room, if the reflex played."""
    by = {name: {r["room"]: r for r in rows if r["agent"] == name} for name in dict.fromkeys(r["agent"] for r in rows)}
    out = {}
    for name, mine in by.items():
        scores = np.array([r["score"] for r in mine.values()])
        out[name] = {"n": len(scores), "mean": round(float(scores.mean()), 4), "deaths": sum(r["died"] for r in mine.values())}
        if "reflex" in by:
            diff = np.array([mine[k]["score"] - by["reflex"][k]["score"] for k in mine])
            lo, hi = paired_interval(diff)
            out[name].update(gap_to_reflex=round(float(diff.mean()), 4), gap_interval=[round(lo, 4), round(hi, 4)],
                             ahead_level_behind=[int((diff > 0.005).sum()), int((abs(diff) <= 0.005).sum()), int((diff < -0.005).sum())])
    return out


def write(folder: Path, rows: list, head: dict) -> None:
    """rows.csv and summary.json in `folder`. A folder that already holds a run is never written over."""
    if (folder / "rows.csv").exists():
        raise FileExistsError(f"{folder} already holds a run (rows.csv); name another folder")
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "rows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (folder / "summary.json").write_text(json.dumps({"provisional": True, **head, "agents": summary(rows)}, indent=1), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Play the fire task and write the scores. Provisional.")
    ap.add_argument("--agents", nargs="*", default=["reflex"], choices=AGENTS)
    ap.add_argument("--program", default=None, help="a program file (PROGRAM_API.md); it plays beside the agents")
    ap.add_argument("--first", type=int, default=0, help="first room")
    ap.add_argument("--rooms", type=int, default=5, help="number of rooms")
    ap.add_argument("--items", type=int, default=8)
    ap.add_argument("--values", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--out", required=True, help="the folder to write rows.csv and summary.json into")
    args = ap.parse_args(argv)
    folder = Path(args.out)
    if (folder / "rows.csv").exists():  # said before any room is played, not after
        ap.error(f"{folder} already holds a run (rows.csv); name another folder")
    if not args.agents and not args.program:
        ap.error("nothing to play: give --agents or --program")
    rows = play(args.agents, range(args.first, args.first + args.rooms), args.items, tuple(args.values), args.program)
    head = {"first": args.first, "rooms": args.rooms, "items": args.items, "values": args.values, "program": args.program}
    write(folder, rows, head)
    print(json.dumps(summary(rows), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
