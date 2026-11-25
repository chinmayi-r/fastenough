# fastenough

**What this is.** Tasks for a robot where the world keeps moving while the robot thinks, so every second of
thinking has a price. There is one task so far: a fire spreads in the lab room, and the robot carries items to a
safe strip before the room burns. It runs on the lab's kinder simulator (kindergarden at commit `86c55a4`, pybullet).

**What it is for.** To study when thinking is worth its time. A simple rule that never thinks and a strong model
that thinks slowly should both fall short, so that there is something to gain from choosing between them.

**Where things are.**

| folder | what is in it |
|---|---|
| [fastenough/tasks/fire/](fastenough/tasks/fire/) | the fire task: the rules, the room, the commands, the score, the baselines, and its tests |
| [fastenough/design/](fastenough/design/) | the tools used to choose the task's settings |
| [fastenough/research/fire/](fastenough/research/fire/) | experiments on the fire task, their scripts and every run |
| [llm_notes/](llm_notes/) | working notes of the coding sessions: plan, design, log of runs, numbered assumptions |
| [scripts/](scripts/) | setting up the simulator |

`tasks` imports nothing from `design` or `research`, and `design` imports nothing from `research`.
[fastenough/test_layers.py](fastenough/test_layers.py) checks this.

## The fire room

- A grid of 16 columns by 15 rows; one cell is 0.2 m; one tick is 0.1 s.
- One partition across row 5 with one doorway. Rows 11 and above are the safe strip.
- 8 items with values 1, 2, 3, 5 or 8. An item on a burning cell loses 2 % of its value per tick until it is picked.
- The robot has health 100 and loses 2 per tick while its cell burns.
- An episode ends when half the floor burns, after 500 ticks, or at health 0.
- Score = value banked on the safe strip / value on the floor at the start.
- A room is named by a number (its seed). The same number is the same room every time.

The room exists twice, room for room:

| name | file | what it is |
|---|---|---|
| the rule check | [gate.py](fastenough/tasks/fire/gate.py) | a small board model of the rules, fast, used to choose the rule set |
| the simulator | [env.py](fastenough/tasks/fire/env.py) | the actual task on kinder and pybullet |

## How to run it

Set up the simulator once with [scripts/della_firelab_setup.sh](scripts/della_firelab_setup.sh). Then, from the top
of the repository with `PYTHONPATH` set to it:

```
python -m pytest -q fastenough
python -m fastenough.tasks.fire.validate --program fastenough/tasks/fire/examples/reflex_program.py --first 50 --rooms 5 --out my_program.json
python -m fastenough.research.fire.run_rooms --tag my_run --agents random reflex --first 0 --rooms 5
```

- The second line plays a program and the reflex on the same rooms and writes the scores to the file you name.
- The third line is the research runner. A run named `my_run` writes into
  `fastenough/research/fire/results/firelab_sim/my_run/`. `--help` lists every setting.

## How to add an agent

Write a program: one Python file with a class `Policy` that has `decide(obs)`. What it sees and what it may answer
is in [PROGRAM_API.md](fastenough/tasks/fire/PROGRAM_API.md). A worked example is
[examples/reflex_program.py](fastenough/tasks/fire/examples/reflex_program.py). Play it with the second line above.

## The agents that exist

| name | where | what it does |
|---|---|---|
| `random`, `reflex` | [tasks/fire/agents.py](fastenough/tasks/fire/agents.py) | random commands; a simple rule that gets an item, brings it home and never enters fire |
| `always`, `commit`, `moving` | [tasks/fire/thinkers.py](fastenough/tasks/fire/thinkers.py) | a planner before every command; a planner that plans once; the reflex acting while the planner plans |
| `model`, `rrara1` | [research/fire/model_agents.py](fastenough/research/fire/model_agents.py) | a language model asked before every command; the rule acting while a model checks it in the background |
| a program | [tasks/fire/programs.py](fastenough/tasks/fire/programs.py) | a Python file with no model call |

Thinking time is charged in ticks, ticks = ceil(seconds x 10) ([clock.py](fastenough/tasks/fire/clock.py)).

## Where the results are

| folder | what is in it |
|---|---|
| [results/firelab_sim/](fastenough/research/fire/results/firelab_sim/) | the simulator runs. One folder per run, one row per room and agent in `rows_*.csv`. The saved games are not in git. |
| [results/firelab_gate/](fastenough/research/fire/results/firelab_gate/) | the rule check's files |
| [results/firelab_compiled/](fastenough/research/fire/results/firelab_compiled/) | programs written before the game by a model, and their test files |

Scripts in [fastenough/research/fire/scripts/](fastenough/research/fire/scripts/) read these files and play nothing:
`all_agents_table.py` (every agent on rooms 0 to 49), `gap_table.py` (every run against the reflex),
`liane_summary.py` (a program on rooms 50 to 99 beside the other agents), `run_summary.py` and
`model_run_summary.py` (one run in detail).

The earlier work on the HAZARD benchmark is in a separate repository, `thinking-fast-hazard`.
