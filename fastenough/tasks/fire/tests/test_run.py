"""Tests for fastenough.tasks.fire.run, the provisional plain runner (PLAN.md section 10).

Rooms 3 and 4 with 6 items are played by the three agents and the example program; then the reflex and the planner
are called directly in the same rooms to compare. Three rooms are built in all (about a minute).
Run: venv_kinder\\Scripts\\python.exe -m pytest -p no:cacheprovider -q fastenough/tasks/fire/tests/test_run.py
"""

import csv
import json
from pathlib import Path

import pytest

from fastenough.tasks.fire.agents import reflex
from fastenough.tasks.fire.clock import Clock
from fastenough.tasks.fire.env import FireLab3DEnv
from fastenough.tasks.fire.run import COLUMNS, main, play, summary
from fastenough.tasks.fire.skills import Skills
from fastenough.tasks.fire.thinkers import thinker

EXAMPLE = Path(__file__).parents[1] / "examples" / "reflex_program.py"
ROOMS = [3, 4]


@pytest.fixture(scope="module")
def rows():
    return play(["random", "reflex", "planner"], ROOMS, items=6, program=EXAMPLE)


def test_one_row_per_room_and_agent_with_every_column(rows):
    assert [(r["room"], r["agent"]) for r in rows] == [(room, a) for room in ROOMS for a in ("random", "reflex", "planner", "program")]
    assert all(list(r) == COLUMNS and 0.0 <= r["score"] <= 1.0 for r in rows)
    assert all(r["decide_s"] == "" for r in rows if r["agent"] in ("random", "reflex"))  # not measured, so not claimed
    assert all(r["decide_s"] > 0 for r in rows if r["agent"] == "planner") and all(r["decide_s"] != "" for r in rows if r["agent"] == "program")


def test_each_agent_scores_as_when_it_is_called_directly(rows):
    env = FireLab3DEnv(level=2, num_items=6, values=(1, 2, 3, 5, 8), realistic_bg=False)
    skills = Skills(env)
    try:
        for room in ROOMS:
            got = {r["agent"]: r for r in rows if r["room"] == room}
            env.reset(seed=room)
            skills.start()
            reflex(skills)
            assert (got["reflex"]["score"], got["reflex"]["ticks"]) == (round(skills.score(), 4), skills.ticks)
            # the example program is the reflex written as a program: same score, and nothing illegal
            assert got["program"]["score"] == got["reflex"]["score"] and got["program"]["illegal"] == 0 and got["program"]["raised"] == 0
            env.reset(seed=room)
            skills.start()
            thinker(skills, Clock())
            assert (got["planner"]["score"], got["planner"]["ticks"]) == (round(skills.score(), 4), skills.ticks)
    finally:
        env.close()


def test_the_summary_gives_each_agent_its_mean_and_its_gap_to_the_reflex(rows):
    s = summary(rows)
    assert list(s) == ["random", "reflex", "planner", "program"] and all(s[a]["n"] == 2 for a in s)
    for name in s:
        scores = [r["score"] for r in rows if r["agent"] == name]
        base = [r["score"] for r in rows if r["agent"] == "reflex"]
        assert s[name]["mean"] == round(sum(scores) / 2, 4)
        assert s[name]["gap_to_reflex"] == pytest.approx(sum(a - b for a, b in zip(scores, base)) / 2, abs=1e-4)
    assert s["reflex"]["gap_to_reflex"] == 0.0 and s["program"]["ahead_level_behind"] == [0, 2, 0]
    assert "gap_to_reflex" not in summary([r for r in rows if r["agent"] == "planner"])["planner"]  # no reflex, no gap


def test_the_command_writes_both_files_and_never_writes_over_a_run(tmp_path):
    out = tmp_path / "my_run"
    args = ["--agents", "reflex", "--first", "3", "--rooms", "1", "--items", "6", "--out", str(out)]
    assert main(args) == 0
    written = list(csv.DictReader((out / "rows.csv").open(newline="")))
    assert [(r["room"], r["agent"]) for r in written] == [("3", "reflex")] and list(written[0]) == COLUMNS
    record = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert record["provisional"] is True and record["agents"]["reflex"]["n"] == 1 and record["items"] == 6
    before = (out / "rows.csv").read_text()
    with pytest.raises(SystemExit):
        main(args)  # the same folder again: refused before any room is played
    assert (out / "rows.csv").read_text() == before
    with pytest.raises(SystemExit):
        main(["--agents", "--out", str(tmp_path / "empty")])  # no agent and no program
