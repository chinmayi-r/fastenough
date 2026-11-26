# The API a program plays through

# The program API for the fire task, version 1

This is what a program-writer reads before writing a program that plays the fire task with no model call.
Everything here is backed by `firelab/interface.py` (the observation and the commands) and `firelab/programs.py`
(the loader and the runner). The worked example is `firelab/examples/reflex_program.py`.

## 1. What a program is

A Python module that defines

```python
class Policy:
    def __init__(self, rules):   # rules: the parts of the observation that never change (section 3, "rules")
        ...
    def decide(self, obs):       # obs: the dictionary of section 3
        return ("get", 3)        # one command of section 4, or None to stop for good
```

- It may import only: math, collections, heapq, itertools, functools, typing, dataclasses, numpy. No files, network,
  model clients, clock or randomness. A file that breaks this is refused before it plays.
- `decide` may keep memory in `self`. A new `Policy` is made for every episode.
- If `decide` raises, or returns a command that is illegal, one tick is lost, the fact is recorded, and the episode
  goes on. A program must never crash: every branch returns a command or None.
- Returning None ends the program's part in the episode; what is banked is kept.

## 2. The episode

- A room with a fire that starts in one cell and spreads. The floor is a grid of 16 columns (i, 0 to 15) by 15 rows
  (j, 0 to 14); a cell is 0.2 m. One partition runs across row 5 with a doorway; rows 11 and above are the safe strip.
- Items lie on the floor. Score = value banked / value on the floor at the start. Only banked items count.
- To bank: pick items into the bag, walk to the safe strip, drop. A drop banks the whole bag at once.
- An item on a burning cell loses 2 % of its starting value per tick until picked. A picked item keeps its value.
- The robot has health 100 and loses 2 per tick while its cell burns. At health 0 the episode ends and the bag is lost.
- The episode also ends when half the floor burns, or after 500 ticks. Whatever is in the bag then is lost.
- Time is in ticks (0.1 s). Walking costs 4 ticks per cell, a pick 10 ticks, a drop 5 ticks.

## 3. The observation (`Session.observe()`), plain numbers, strings and lists

| key | meaning |
|---|---|
| `tick`, `ticks_left` | time now, and ticks until the 500-tick cap |
| `health`, `max_health` | the robot's health |
| `burning_share`, `end_share` | share of the floor burning now; the share that ends the episode (0.5) |
| `robot` | `{"cell": [i, j], "x", "y", "in_safe": bool, "in_fire": bool}` |
| `bag`, `bag_value` | ids of the items carried, and what they are worth together |
| `banked_value`, `total_value`, `score` | banked so far; value at the start; banked / total |
| `items` | one record per item, in id order (below) |
| `home` | `{"cells_away", "ticks"}`: a fire-free route to the safe strip and the ticks to walk it and bank; both None if there is none |
| `grid` | `{"nx": 16, "ny": 15, "cell_m": 0.2, "burning": [[i, j], ...], "walls": [[i, j], ...], "safe_rows_from": 11, "open": [[i, j], ...]}`; `open` lists the cells the robot can stand on |
| `job` | None, or the pick or drop under way |
| `last_command`, `last_result` | the previous command as a list, and `{"ticks", "done", "reason"}`; both None at the start |
| `costs` | `{"ticks_per_cell": 4, "pick_ticks": 10, "drop_ticks": 5, "item_loss_per_tick": 0.02, "health_loss_per_tick": 2.0}` |

An item record: `{"id", "cell": [i, j], "value" (now), "start" (at the beginning), "where": "floor" | "bag" | "saved" | "lost",
"burning": bool, "cells_away", "ticks_to_get"}`. `cells_away` and `ticks_to_get` describe a fire-free route to a cell from
which the item can be picked (its own cell or one of the 8 around it); both are None when there is no such route or
the item is not on the floor.

`rules` (given to `__init__`) holds `api_version`, `costs`, `end_share`, `max_health`, `total_value`.

The whole room is visible: every item and every burning cell. What is NOT visible: where the fire will be later.

## 4. Commands

| command | what the robot does | result when it cannot |
|---|---|---|
| `("get", k)` | walks by a fire-free route into reach of item k and picks it | "no route" (0 ticks if there is none from the start) |
| `("get", k, True)` | the same, but the route may cross fire (costs health) | "episode over" if it dies on the way |
| `("home",)` | walks by a fire-free route to the safe strip and banks the bag | "no route"; "refused" with an empty bag |
| `("home", True)` | the same, through fire if it must | |
| `("go", (i, j))` / `("go", (i, j), True)` | walks to a cell of `grid["open"]` | "no route" |
| `("pick", k)` / `("pick",)` | picks item k, or any item in reach, without walking | "refused" |
| `("drop",)` | banks the bag here | "refused" outside the safe strip or with an empty bag |
| `("wait", n)` | stands n ticks, 1 to 100 | |

A walk re-plans after every cell, so it adapts to the fire. A command returns only when it is finished or has failed;
`decide` is called again after it. A legal command that cannot be done costs what it used (often nothing).

## 5. What the reflex example does, and where it loses

It fetches the most valuable item left, banks after every pick, never walks through fire, and stops when it cannot
get home. It scores about 0.60 to 0.63 on average. It loses value by walking home after every single pick, by
never crossing fire even when the health cost would be small, and by ignoring how soon an item or a route will burn.


# A worked example that loads and plays (the reflex)

```python
"""Worked example of a program (firelab/programs.py): the official reflex, written against the observation.

It decides exactly as firelab.agents.reflex does (tested in firelab/test_programs.py), so the reflex's scores apply
to it. A compiled program has this shape: one class Policy with __init__(rules) and decide(obs), nothing imported
beyond the allowed list.

Rules, in order, each decision:
  1. the last "home" failed            -> stop (trapped with the load, or out of time)
  2. something is in the bag           -> ("home",)      bank after every pick
  3. items left that are worth fetching -> ("get", k)    the most valuable; the nearer one on a tie
  4. nothing left                       -> stop
An item whose "get" failed is not tried again: the fire only grows.
"""


class Policy:
    def __init__(self, rules):
        self.rules = rules
        self.skip = set()  # items a "get" could not reach

    def decide(self, obs):
        last, result = obs["last_command"], obs["last_result"]
        if last is not None and not result["done"]:
            if last[0] == "home":
                return None
            if last[0] == "get":
                self.skip.add(last[1])
        if obs["bag"]:
            return ("home",)
        todo = [i for i in obs["items"] if i["where"] == "floor" and i["value"] > 0 and i["id"] not in self.skip]
        if not todo:
            return None
        x, y = obs["robot"]["cell"]
        far = lambda i: abs(i["cell"][0] - x) + abs(i["cell"][1] - y)
        best = max(todo, key=lambda i: (i["value"], -far(i)))
        return ("get", best["id"])
```

# One complete text as the slow agent saw it (for the layout of the room)

You control a robot in a room where a fire is spreading. Items lie on the floor. Carry them to the safe strip and bank them there: only banked items count, each at the value it has when you pick it up.

Time: tick 0; at most 500 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 0 % burns now.
You: cell (9, 7), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 0.0 of 32.0.
Costs: walking 4 ticks per cell, a pick 10 ticks, banking 5 ticks. An item on a burning cell loses 2 % of its starting value per tick. In fire you lose 2 health per tick.

Items on the floor:
  item 0 at (8, 6): worth 1.0 (started at 1); 0 cells away, about 10 ticks to walk there and pick it
  item 1 at (12, 6): worth 1.0 (started at 1); 2 cells away, about 18 ticks to walk there and pick it
  item 2 at (15, 3): worth 5.0 (started at 5); 11 cells away, about 54 ticks to walk there and pick it
  item 3 at (4, 9): worth 8.0 (started at 8); 5 cells away, about 30 ticks to walk there and pick it
  item 4 at (13, 7): worth 5.0 (started at 5); 3 cells away, about 22 ticks to walk there and pick it
  item 5 at (3, 7): worth 3.0 (started at 3); 5 cells away, about 30 ticks to walk there and pick it
  item 6 at (7, 9): worth 1.0 (started at 1); 2 cells away, about 18 ticks to walk there and pick it
  item 7 at (15, 2): worth 8.0 (started at 8); 11 cells away, about 54 ticks to walk there and pick it
Safe strip (rows 11 and above): 4 cells away, about 21 ticks to walk there and bank

What you did so far:
  nothing yet

Map (the number on the left is the row; the top row is row 14, columns run 0 to 15 left to right; # wall or furniture, F fire, = safe strip, . floor, R you, a digit is that item):
14 ==##############
13 ==#####=========
12 ================
11 ================
10 ................
 9 ....3..6........
 8 ................
 7 ...5.....R...4..
 6 ........0...1F..
 5 ######....######
 4 ................
 3 ...............2
 2 ...............7
 1 ................
 0 ................

Options:
A. get item 3 (worth 8.0 now; about 30 ticks)
B. get item 7 (worth 8.0 now; about 54 ticks)
C. get item 2 (worth 5.0 now; about 54 ticks)
D. get item 4 (worth 5.0 now; about 22 ticks)
E. get item 5 (worth 3.0 now; about 30 ticks)
F. get item 0 (worth 1.0 now; about 10 ticks)
G. get item 1 (worth 1.0 now; about 18 ticks)
H. get item 6 (worth 1.0 now; about 18 ticks)
I. wait 10 ticks

# Recorded games of the slow agent on training rooms (what it saw, what it chose, what happened)

## Room 0: final score 0.625, ended at tick 345 with health 0; 9 decisions

Decision 1
Time: tick 0; at most 500 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 0 % burns now.
You: cell (9, 7), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 0.0 of 32.0.
Items on the floor:
  item 0 at (8, 6): worth 1.0 (started at 1); 0 cells away, about 10 ticks to walk there and pick it
  item 1 at (12, 6): worth 1.0 (started at 1); 2 cells away, about 18 ticks to walk there and pick it
  item 2 at (15, 3): worth 5.0 (started at 5); 11 cells away, about 54 ticks to walk there and pick it
  item 3 at (4, 9): worth 8.0 (started at 8); 5 cells away, about 30 ticks to walk there and pick it
  item 4 at (13, 7): worth 5.0 (started at 5); 3 cells away, about 22 ticks to walk there and pick it
  item 5 at (3, 7): worth 3.0 (started at 3); 5 cells away, about 30 ticks to walk there and pick it
  item 6 at (7, 9): worth 1.0 (started at 1); 2 cells away, about 18 ticks to walk there and pick it
  item 7 at (15, 2): worth 8.0 (started at 8); 11 cells away, about 54 ticks to walk there and pick it
Safe strip (rows 11 and above): 4 cells away, about 21 ticks to walk there and bank
Chose: get item 3 (worth 8.0 now; about 30 ticks), after 0.2 s of thinking (3 ticks standing)
Result: ok, 28 ticks

Decision 2
Time: tick 31; at most 469 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 2 % burns now.
You: cell (5, 8), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 8.0, not banked yet. Banked so far: 0.0 of 32.0.
Items on the floor:
  item 0 at (8, 6): worth 1.0 (started at 1); 3 cells away, about 22 ticks to walk there and pick it
  item 1 at (12, 6): worth 1.0 (started at 1); 7 cells away, about 38 ticks to walk there and pick it
  item 2 at (15, 3): worth 5.0 (started at 5); 14 cells away, about 66 ticks to walk there and pick it
  item 4 at (13, 7): worth 3.8 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 38 ticks; 7 cells away, about 38 ticks to walk there and pick it
  item 5 at (3, 7): worth 3.0 (started at 3); 1 cells away, about 14 ticks to walk there and pick it
  item 6 at (7, 9): worth 1.0 (started at 1); 1 cells away, about 14 ticks to walk there and pick it
  item 7 at (15, 2): worth 8.0 (started at 8); 14 cells away, about 66 ticks to walk there and pick it
Safe strip (rows 11 and above): 3 cells away, about 17 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 8.0; about 17 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 17 ticks

Decision 3
Time: tick 50; at most 450 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 3 % burns now.
You: cell (5, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 8.0 of 32.0.
Items on the floor:
  item 0 at (8, 6): worth 1.0 (started at 1); 6 cells away, about 34 ticks to walk there and pick it
  item 1 at (12, 6): worth 1.0 (started at 1); 10 cells away, about 50 ticks to walk there and pick it
  item 2 at (15, 3): worth 5.0 (started at 5); 17 cells away, about 78 ticks to walk there and pick it
  item 4 at (13, 7): worth 1.9 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 19 ticks; 10 cells away, about 50 ticks to walk there and pick it
  item 5 at (3, 7): worth 3.0 (started at 3); 4 cells away, about 26 ticks to walk there and pick it
  item 6 at (7, 9): worth 1.0 (started at 1); 2 cells away, about 18 ticks to walk there and pick it
  item 7 at (15, 2): worth 8.0 (started at 8); 17 cells away, about 78 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 7 (worth 8.0 now; about 78 ticks), after 0.2 s of thinking (2 ticks standing)
Result: ok, 78 ticks

Decision 4
Time: tick 130; at most 370 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 11 % burns now.
You: cell (14, 3), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 8.0, not banked yet. Banked so far: 8.0 of 32.0.
Items on the floor:
  item 0 at (8, 6): worth 1.0 (started at 1); 8 cells away, about 42 ticks to walk there and pick it
  item 1 at (12, 6): worth 1.0 (started at 1); no fire-free route to it
  item 2 at (15, 3): worth 5.0 (started at 5); 0 cells away, about 10 ticks to walk there and pick it
  item 4 at (13, 7): worth 0.0 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 5 at (3, 7): worth 3.0 (started at 3); 14 cells away, about 66 ticks to walk there and pick it
  item 6 at (7, 9): worth 1.0 (started at 1); 11 cells away, about 54 ticks to walk there and pick it
Safe strip (rows 11 and above): 14 cells away, about 61 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 8.0; about 61 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 65 ticks

Decision 5
Time: tick 197; at most 303 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 17 % burns now.
You: cell (7, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 16.0 of 32.0.
Items on the floor:
  item 0 at (8, 6): worth 1.0 (started at 1); 4 cells away, about 26 ticks to walk there and pick it
  item 1 at (12, 6): worth 0.5 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 23 ticks; no fire-free route to it
  item 2 at (15, 3): worth 5.0 (started at 5); no fire-free route to it
  item 4 at (13, 7): worth 0.0 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 5 at (3, 7): worth 3.0 (started at 3); 6 cells away, about 34 ticks to walk there and pick it
  item 6 at (7, 9): worth 1.0 (started at 1); 1 cells away, about 14 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 5 (worth 3.0 now; about 34 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 34 ticks

Decision 6
Time: tick 233; at most 267 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 22 % burns now.
You: cell (4, 8), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 3.0, not banked yet. Banked so far: 16.0 of 32.0.
Items on the floor:
  item 0 at (8, 6): worth 1.0 (started at 1); 4 cells away, about 26 ticks to walk there and pick it
  item 1 at (12, 6): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 2 at (15, 3): worth 5.0 (started at 5); no fire-free route to it
  item 4 at (13, 7): worth 0.0 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 6 at (7, 9): worth 1.0 (started at 1); 2 cells away, about 18 ticks to walk there and pick it
Safe strip (rows 11 and above): 3 cells away, about 17 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 3.0; about 17 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 17 ticks

Decision 7
Time: tick 252; at most 248 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 27 % burns now.
You: cell (4, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 19.0 of 32.0.
Items on the floor:
  item 0 at (8, 6): worth 1.0 (started at 1); 8 cells away, about 42 ticks to walk there and pick it
  item 1 at (12, 6): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 2 at (15, 3): worth 5.0 (started at 5); no fire-free route to it
  item 4 at (13, 7): worth 0.0 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 6 at (7, 9): worth 1.0 (started at 1); 3 cells away, about 22 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 6 (worth 1.0 now; about 22 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 22 ticks

Decision 8
Time: tick 276; at most 224 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 32 % burns now.
You: cell (6, 10), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 1.0, not banked yet. Banked so far: 19.0 of 32.0.
Items on the floor:
  item 0 at (8, 6): worth 1.0 (started at 1); no fire-free route to it
  item 1 at (12, 6): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 2 at (15, 3): worth 5.0 (started at 5); no fire-free route to it
  item 4 at (13, 7): worth 0.0 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 0 ticks; no fire-free route to it
Safe strip (rows 11 and above): 1 cells away, about 9 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 1.0; about 9 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 9 ticks

Decision 9
Time: tick 287; at most 213 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 35 % burns now.
You: cell (6, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 20.0 of 32.0.
Items on the floor:
  item 0 at (8, 6): worth 0.9 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 44 ticks; no fire-free route to it
  item 1 at (12, 6): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 2 at (15, 3): worth 5.0 (started at 5); no fire-free route to it
  item 4 at (13, 7): worth 0.0 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 0 ticks; no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 2, walking through fire (worth 5.0 now; no fire-free route, time unknown), after 0.1 s of thinking (2 ticks standing)
Result: episode over, 56 ticks

## Room 1: final score 0.619, ended at tick 250 with health 0; 11 decisions

Decision 1
Time: tick 0; at most 500 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 0 % burns now.
You: cell (9, 7), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 0.0 of 42.0.
Items on the floor:
  item 0 at (8, 1): worth 3.0 (started at 3); 6 cells away, about 34 ticks to walk there and pick it
  item 1 at (5, 1): worth 3.0 (started at 3); 8 cells away, about 42 ticks to walk there and pick it
  item 2 at (9, 6): worth 8.0 (started at 8); 0 cells away, about 10 ticks to walk there and pick it
  item 3 at (9, 4): worth 8.0 (started at 8); 3 cells away, about 22 ticks to walk there and pick it
  item 4 at (7, 10): worth 8.0 (started at 8); 3 cells away, about 22 ticks to walk there and pick it
  item 5 at (5, 7): worth 5.0 (started at 5); 3 cells away, about 22 ticks to walk there and pick it
  item 6 at (8, 0): worth 2.0 (started at 2); 7 cells away, about 38 ticks to walk there and pick it
  item 7 at (2, 6): worth 5.0 (started at 5); 6 cells away, about 34 ticks to walk there and pick it
Safe strip (rows 11 and above): 4 cells away, about 21 ticks to walk there and bank
Chose: get item 2 (worth 8.0 now; about 10 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 10 ticks

Decision 2
Time: tick 12; at most 488 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 1 % burns now.
You: cell (9, 7), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 8.0, not banked yet. Banked so far: 0.0 of 42.0.
Items on the floor:
  item 0 at (8, 1): worth 3.0 (started at 3); 6 cells away, about 34 ticks to walk there and pick it
  item 1 at (5, 1): worth 3.0 (started at 3); 8 cells away, about 42 ticks to walk there and pick it
  item 3 at (9, 4): worth 8.0 (started at 8); 3 cells away, about 22 ticks to walk there and pick it
  item 4 at (7, 10): worth 8.0 (started at 8); 3 cells away, about 22 ticks to walk there and pick it
  item 5 at (5, 7): worth 5.0 (started at 5); 4 cells away, about 26 ticks to walk there and pick it
  item 6 at (8, 0): worth 2.0 (started at 2); 7 cells away, about 38 ticks to walk there and pick it
  item 7 at (2, 6): worth 5.0 (started at 5); 8 cells away, about 42 ticks to walk there and pick it
Safe strip (rows 11 and above): 4 cells away, about 21 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 8.0; about 21 ticks), after 0.1 s of thinking (1 ticks standing)
Result: ok, 21 ticks

Decision 3
Time: tick 34; at most 466 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 1 % burns now.
You: cell (9, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 8.0 of 42.0.
Items on the floor:
  item 0 at (8, 1): worth 3.0 (started at 3); 10 cells away, about 50 ticks to walk there and pick it
  item 1 at (5, 1): worth 3.0 (started at 3); 12 cells away, about 58 ticks to walk there and pick it
  item 3 at (9, 4): worth 8.0 (started at 8); 7 cells away, about 38 ticks to walk there and pick it
  item 4 at (7, 10): worth 8.0 (started at 8); 1 cells away, about 14 ticks to walk there and pick it
  item 5 at (5, 7): worth 5.0 (started at 5); 6 cells away, about 34 ticks to walk there and pick it
  item 6 at (8, 0): worth 2.0 (started at 2); 11 cells away, about 54 ticks to walk there and pick it
  item 7 at (2, 6): worth 5.0 (started at 5); 10 cells away, about 50 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 4 (worth 8.0 now; about 14 ticks), after 0.1 s of thinking (1 ticks standing)
Result: ok, 14 ticks

Decision 4
Time: tick 49; at most 451 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 1 % burns now.
You: cell (8, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 8.0, not banked yet. Banked so far: 8.0 of 42.0.
Items on the floor:
  item 0 at (8, 1): worth 3.0 (started at 3); 9 cells away, about 46 ticks to walk there and pick it
  item 1 at (5, 1): worth 3.0 (started at 3); 11 cells away, about 54 ticks to walk there and pick it
  item 3 at (9, 4): worth 8.0 (started at 8); 6 cells away, about 34 ticks to walk there and pick it
  item 5 at (5, 7): worth 5.0 (started at 5); 5 cells away, about 30 ticks to walk there and pick it
  item 6 at (8, 0): worth 2.0 (started at 2); 10 cells away, about 50 ticks to walk there and pick it
  item 7 at (2, 6): worth 5.0 (started at 5); 9 cells away, about 46 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 8.0; about 5 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 5 ticks

Decision 5
Time: tick 56; at most 444 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 1 % burns now.
You: cell (8, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 16.0 of 42.0.
Items on the floor:
  item 0 at (8, 1): worth 3.0 (started at 3); 9 cells away, about 46 ticks to walk there and pick it
  item 1 at (5, 1): worth 3.0 (started at 3); 11 cells away, about 54 ticks to walk there and pick it
  item 3 at (9, 4): worth 8.0 (started at 8); 6 cells away, about 34 ticks to walk there and pick it
  item 5 at (5, 7): worth 5.0 (started at 5); 5 cells away, about 30 ticks to walk there and pick it
  item 6 at (8, 0): worth 2.0 (started at 2); 10 cells away, about 50 ticks to walk there and pick it
  item 7 at (2, 6): worth 5.0 (started at 5); 9 cells away, about 46 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 3 (worth 8.0 now; about 34 ticks), after 0.1 s of thinking (2 ticks standing)
Result: no route, 8 ticks

Decision 6
Time: tick 66; at most 434 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 1 % burns now.
You: cell (8, 9), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 16.0 of 42.0.
Items on the floor:
  item 0 at (8, 1): worth 3.0 (started at 3); no fire-free route to it
  item 1 at (5, 1): worth 3.0 (started at 3); no fire-free route to it
  item 3 at (9, 4): worth 8.0 (started at 8); no fire-free route to it
  item 5 at (5, 7): worth 5.0 (started at 5); 3 cells away, about 22 ticks to walk there and pick it
  item 6 at (8, 0): worth 2.0 (started at 2); no fire-free route to it
  item 7 at (2, 6): worth 5.0 (started at 5); 7 cells away, about 38 ticks to walk there and pick it
Safe strip (rows 11 and above): 2 cells away, about 13 ticks to walk there and bank
Chose: get item 5 (worth 5.0 now; about 22 ticks), after 0.3 s of thinking (4 ticks standing)
Result: ok, 22 ticks

Decision 7
Time: tick 92; at most 408 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 1 % burns now.
You: cell (6, 8), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 5.0, not banked yet. Banked so far: 16.0 of 42.0.
Items on the floor:
  item 0 at (8, 1): worth 3.0 (started at 3); no fire-free route to it
  item 1 at (5, 1): worth 3.0 (started at 3); no fire-free route to it
  item 3 at (9, 4): worth 8.0 (started at 8); no fire-free route to it
  item 6 at (8, 0): worth 2.0 (started at 2); no fire-free route to it
  item 7 at (2, 6): worth 5.0 (started at 5); 4 cells away, about 26 ticks to walk there and pick it
Safe strip (rows 11 and above): 3 cells away, about 17 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 5.0; about 17 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 17 ticks

Decision 8
Time: tick 111; at most 389 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 2 % burns now.
You: cell (6, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 21.0 of 42.0.
Items on the floor:
  item 0 at (8, 1): worth 3.0 (started at 3); no fire-free route to it
  item 1 at (5, 1): worth 3.0 (started at 3); no fire-free route to it
  item 3 at (9, 4): worth 8.0 (started at 8); no fire-free route to it
  item 6 at (8, 0): worth 2.0 (started at 2); no fire-free route to it
  item 7 at (2, 6): worth 5.0 (started at 5); 7 cells away, about 38 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 7 (worth 5.0 now; about 38 ticks), after 0.1 s of thinking (1 ticks standing)
Result: ok, 38 ticks

Decision 9
Time: tick 150; at most 350 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 8 % burns now.
You: cell (3, 7), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 5.0, not banked yet. Banked so far: 21.0 of 42.0.
Items on the floor:
  item 0 at (8, 1): worth 3.0 (started at 3); no fire-free route to it
  item 1 at (5, 1): worth 3.0 (started at 3); no fire-free route to it
  item 3 at (9, 4): worth 8.0 (started at 8); no fire-free route to it
  item 6 at (8, 0): worth 2.0 (started at 2); no fire-free route to it
Safe strip (rows 11 and above): 4 cells away, about 21 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 5.0; about 21 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 21 ticks

Decision 10
Time: tick 173; at most 327 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 12 % burns now.
You: cell (3, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 26.0 of 42.0.
Items on the floor:
  item 0 at (8, 1): worth 3.0 (started at 3); no fire-free route to it
  item 1 at (5, 1): worth 3.0 (started at 3); no fire-free route to it
  item 3 at (9, 4): worth 8.0 (started at 8); no fire-free route to it
  item 6 at (8, 0): worth 2.0 (started at 2); no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 3, walking through fire (worth 8.0 now; no fire-free route, time unknown), after 0.1 s of thinking (2 ticks standing)
Result: ok, 54 ticks

Decision 11
Time: tick 229; at most 271 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 28 % burns now.
You: cell (8, 5), health 42 of 100. You are standing IN FIRE and lose 2 health per tick. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 2.4, not banked yet. Banked so far: 26.0 of 42.0.
Items on the floor:
  item 0 at (8, 1): worth 3.0 (started at 3); no fire-free route to it
  item 1 at (5, 1): worth 3.0 (started at 3); no fire-free route to it
  item 6 at (8, 0): worth 2.0 (started at 2); no fire-free route to it
Safe strip (rows 11 and above): no fire-free route to it
Chose: go to the safe strip, walking through fire and bank the bag (worth 2.4; no fire-free route, time unknown), after 0.1 s of thinking (2 ticks standing)
Result: episode over, 19 ticks

## Room 2: final score 0.708, ended at tick 366 with health 100; 11 decisions

Decision 1
Time: tick 0; at most 500 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 0 % burns now.
You: cell (9, 7), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 0.0 of 24.0.
Items on the floor:
  item 0 at (2, 1): worth 5.0 (started at 5); 11 cells away, about 54 ticks to walk there and pick it
  item 1 at (13, 2): worth 5.0 (started at 5); 9 cells away, about 46 ticks to walk there and pick it
  item 2 at (3, 2): worth 1.0 (started at 1); 9 cells away, about 46 ticks to walk there and pick it
  item 3 at (13, 8): worth 2.0 (started at 2); 3 cells away, about 22 ticks to walk there and pick it
  item 4 at (4, 8): worth 1.0 (started at 1); 4 cells away, about 26 ticks to walk there and pick it
  item 5 at (0, 7): worth 1.0 (started at 1); 8 cells away, about 42 ticks to walk there and pick it
  item 6 at (4, 7): worth 8.0 (started at 8); 4 cells away, about 26 ticks to walk there and pick it
  item 7 at (15, 4): worth 1.0 (started at 1); 13 cells away, about 62 ticks to walk there and pick it
Safe strip (rows 11 and above): 4 cells away, about 21 ticks to walk there and bank
Chose: get item 6 (worth 8.0 now; about 26 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 24 ticks

Decision 2
Time: tick 26; at most 474 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 1 % burns now.
You: cell (5, 7), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 8.0, not banked yet. Banked so far: 0.0 of 24.0.
Items on the floor:
  item 0 at (2, 1): worth 5.0 (started at 5); 11 cells away, about 54 ticks to walk there and pick it
  item 1 at (13, 2): worth 5.0 (started at 5); 11 cells away, about 54 ticks to walk there and pick it
  item 2 at (3, 2): worth 1.0 (started at 1); 9 cells away, about 46 ticks to walk there and pick it
  item 3 at (13, 8): worth 2.0 (started at 2); 7 cells away, about 38 ticks to walk there and pick it
  item 4 at (4, 8): worth 1.0 (started at 1); 0 cells away, about 10 ticks to walk there and pick it
  item 5 at (0, 7): worth 1.0 (started at 1); 4 cells away, about 26 ticks to walk there and pick it
  item 7 at (15, 4): worth 1.0 (started at 1); 15 cells away, about 70 ticks to walk there and pick it
Safe strip (rows 11 and above): 4 cells away, about 21 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 8.0; about 21 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 21 ticks

Decision 3
Time: tick 49; at most 451 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 2 % burns now.
You: cell (5, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 8.0 of 24.0.
Items on the floor:
  item 0 at (2, 1): worth 5.0 (started at 5); 15 cells away, about 70 ticks to walk there and pick it
  item 1 at (13, 2): worth 5.0 (started at 5); 16 cells away, about 74 ticks to walk there and pick it
  item 2 at (3, 2): worth 1.0 (started at 1); 13 cells away, about 62 ticks to walk there and pick it
  item 3 at (13, 8): worth 2.0 (started at 2); 9 cells away, about 46 ticks to walk there and pick it
  item 4 at (4, 8): worth 1.0 (started at 1); 2 cells away, about 18 ticks to walk there and pick it
  item 5 at (0, 7): worth 1.0 (started at 1); 7 cells away, about 38 ticks to walk there and pick it
  item 7 at (15, 4): worth 1.0 (started at 1); 19 cells away, about 86 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 0 (worth 5.0 now; about 70 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 70 ticks

Decision 4
Time: tick 121; at most 379 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 2 % burns now.
You: cell (3, 2), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 5.0, not banked yet. Banked so far: 8.0 of 24.0.
Items on the floor:
  item 1 at (13, 2): worth 5.0 (started at 5); 10 cells away, about 50 ticks to walk there and pick it
  item 2 at (3, 2): worth 1.0 (started at 1); 0 cells away, about 10 ticks to walk there and pick it
  item 3 at (13, 8): worth 2.0 (started at 2); 14 cells away, about 66 ticks to walk there and pick it
  item 4 at (4, 8): worth 1.0 (started at 1); 11 cells away, about 54 ticks to walk there and pick it
  item 5 at (0, 7): worth 1.0 (started at 1); 15 cells away, about 70 ticks to walk there and pick it
  item 7 at (15, 4): worth 1.0 (started at 1); 14 cells away, about 66 ticks to walk there and pick it
Safe strip (rows 11 and above): 13 cells away, about 57 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 5.0; about 57 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 57 ticks

Decision 5
Time: tick 180; at most 320 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 14 % burns now.
You: cell (7, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 13.0 of 24.0.
Items on the floor:
  item 1 at (13, 2): worth 0.0 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 2 at (3, 2): worth 1.0 (started at 1); 11 cells away, about 54 ticks to walk there and pick it
  item 3 at (13, 8): worth 2.0 (started at 2); 7 cells away, about 38 ticks to walk there and pick it
  item 4 at (4, 8): worth 1.0 (started at 1); 4 cells away, about 26 ticks to walk there and pick it
  item 5 at (0, 7): worth 1.0 (started at 1); 9 cells away, about 46 ticks to walk there and pick it
  item 7 at (15, 4): worth 0.8 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 41 ticks; no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 3 (worth 2.0 now; about 38 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 38 ticks

Decision 6
Time: tick 220; at most 280 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 20 % burns now.
You: cell (12, 9), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 2.0, not banked yet. Banked so far: 13.0 of 24.0.
Items on the floor:
  item 1 at (13, 2): worth 0.0 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 2 at (3, 2): worth 1.0 (started at 1); no fire-free route to it
  item 4 at (4, 8): worth 1.0 (started at 1); 7 cells away, about 38 ticks to walk there and pick it
  item 5 at (0, 7): worth 1.0 (started at 1); 12 cells away, about 58 ticks to walk there and pick it
  item 7 at (15, 4): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 1 ticks; no fire-free route to it
Safe strip (rows 11 and above): 2 cells away, about 13 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 2.0; about 13 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 13 ticks

Decision 7
Time: tick 235; at most 265 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 21 % burns now.
You: cell (12, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 15.0 of 24.0.
Items on the floor:
  item 1 at (13, 2): worth 0.0 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 2 at (3, 2): worth 1.0 (started at 1); no fire-free route to it
  item 4 at (4, 8): worth 1.0 (started at 1); 9 cells away, about 46 ticks to walk there and pick it
  item 5 at (0, 7): worth 1.0 (started at 1); 14 cells away, about 66 ticks to walk there and pick it
  item 7 at (15, 4): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 4 (worth 1.0 now; about 46 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 46 ticks

Decision 8
Time: tick 283; at most 217 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 28 % burns now.
You: cell (5, 9), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 1.0, not banked yet. Banked so far: 15.0 of 24.0.
Items on the floor:
  item 1 at (13, 2): worth 0.0 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 2 at (3, 2): worth 1.0 (started at 1); no fire-free route to it
  item 5 at (0, 7): worth 1.0 (started at 1); 5 cells away, about 30 ticks to walk there and pick it
  item 7 at (15, 4): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
Safe strip (rows 11 and above): 2 cells away, about 13 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 1.0; about 13 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 13 ticks

Decision 9
Time: tick 298; at most 202 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 30 % burns now.
You: cell (5, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 16.0 of 24.0.
Items on the floor:
  item 1 at (13, 2): worth 0.0 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 2 at (3, 2): worth 1.0 (started at 1); no fire-free route to it
  item 5 at (0, 7): worth 1.0 (started at 1); 7 cells away, about 38 ticks to walk there and pick it
  item 7 at (15, 4): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 5 (worth 1.0 now; about 38 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 38 ticks

Decision 10
Time: tick 338; at most 162 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 43 % burns now.
You: cell (1, 8), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 1.0, not banked yet. Banked so far: 16.0 of 24.0.
Items on the floor:
  item 1 at (13, 2): worth 0.0 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 2 at (3, 2): worth 1.0 (started at 1); no fire-free route to it
  item 7 at (15, 4): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
Safe strip (rows 11 and above): 3 cells away, about 17 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 1.0; about 17 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 17 ticks

Decision 11
Time: tick 357; at most 143 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 49 % burns now.
You: cell (1, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 17.0 of 24.0.
Items on the floor:
  item 1 at (13, 2): worth 0.0 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 2 at (3, 2): worth 0.9 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 43 ticks; no fire-free route to it
  item 7 at (15, 4): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: wait 10 ticks, after 0.1 s of thinking (2 ticks standing)
Result: episode over, 7 ticks

## Room 3: final score 0.371, ended at tick 272 with health 76; 8 decisions

Decision 1
Time: tick 0; at most 500 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 0 % burns now.
You: cell (9, 7), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 0.0 of 35.0.
Items on the floor:
  item 0 at (8, 8): worth 3.0 (started at 3); 0 cells away, about 10 ticks to walk there and pick it
  item 1 at (3, 10): worth 8.0 (started at 8); 7 cells away, about 38 ticks to walk there and pick it
  item 2 at (5, 4): worth 5.0 (started at 5); 7 cells away, about 38 ticks to walk there and pick it
  item 3 at (14, 10): worth 5.0 (started at 5); 6 cells away, about 34 ticks to walk there and pick it
  item 4 at (11, 10): worth 5.0 (started at 5); 3 cells away, about 22 ticks to walk there and pick it
  item 5 at (6, 9): worth 1.0 (started at 1); 3 cells away, about 22 ticks to walk there and pick it
  item 6 at (6, 4): worth 3.0 (started at 3); 4 cells away, about 26 ticks to walk there and pick it
  item 7 at (2, 4): worth 5.0 (started at 5); 10 cells away, about 50 ticks to walk there and pick it
Safe strip (rows 11 and above): 4 cells away, about 21 ticks to walk there and bank
Chose: get item 1 (worth 8.0 now; about 38 ticks), after 0.2 s of thinking (2 ticks standing)
Result: ok, 36 ticks

Decision 2
Time: tick 38; at most 462 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 3 % burns now.
You: cell (4, 9), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 8.0, not banked yet. Banked so far: 0.0 of 35.0.
Items on the floor:
  item 0 at (8, 8): worth 3.0 (started at 3); 3 cells away, about 22 ticks to walk there and pick it
  item 2 at (5, 4): worth 5.0 (started at 5); 10 cells away, about 50 ticks to walk there and pick it
  item 3 at (14, 10): worth 5.0 (started at 5); 13 cells away, about 62 ticks to walk there and pick it
  item 4 at (11, 10): worth 2.6 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 26 ticks; 6 cells away, about 34 ticks to walk there and pick it
  item 5 at (6, 9): worth 1.0 (started at 1); 1 cells away, about 14 ticks to walk there and pick it
  item 6 at (6, 4): worth 3.0 (started at 3); 7 cells away, about 38 ticks to walk there and pick it
  item 7 at (2, 4): worth 5.0 (started at 5); 13 cells away, about 62 ticks to walk there and pick it
Safe strip (rows 11 and above): 2 cells away, about 13 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 8.0; about 13 ticks), after 0.2 s of thinking (2 ticks standing)
Result: ok, 13 ticks

Decision 3
Time: tick 53; at most 447 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 4 % burns now.
You: cell (4, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 8.0 of 35.0.
Items on the floor:
  item 0 at (8, 8): worth 3.0 (started at 3); 5 cells away, about 30 ticks to walk there and pick it
  item 2 at (5, 4): worth 5.0 (started at 5); 12 cells away, about 58 ticks to walk there and pick it
  item 3 at (14, 10): worth 5.0 (started at 5); 16 cells away, about 74 ticks to walk there and pick it
  item 4 at (11, 10): worth 1.1 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 11 ticks; 6 cells away, about 34 ticks to walk there and pick it
  item 5 at (6, 9): worth 1.0 (started at 1); 2 cells away, about 18 ticks to walk there and pick it
  item 6 at (6, 4): worth 3.0 (started at 3); 9 cells away, about 46 ticks to walk there and pick it
  item 7 at (2, 4): worth 5.0 (started at 5); 15 cells away, about 70 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 2 (worth 5.0 now; about 58 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 58 ticks

Decision 4
Time: tick 113; at most 387 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 13 % burns now.
You: cell (6, 3), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 5.0, not banked yet. Banked so far: 8.0 of 35.0.
Items on the floor:
  item 0 at (8, 8): worth 3.0 (started at 3); 5 cells away, about 30 ticks to walk there and pick it
  item 3 at (14, 10): worth 5.0 (started at 5); no fire-free route to it
  item 4 at (11, 10): worth 0.0 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 0 ticks; 10 cells away, about 50 ticks to walk there and pick it
  item 5 at (6, 9): worth 1.0 (started at 1); 6 cells away, about 34 ticks to walk there and pick it
  item 6 at (6, 4): worth 3.0 (started at 3); 0 cells away, about 10 ticks to walk there and pick it
  item 7 at (2, 4): worth 5.0 (started at 5); 3 cells away, about 22 ticks to walk there and pick it
Safe strip (rows 11 and above): 9 cells away, about 41 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 5.0; about 41 ticks), after 0.2 s of thinking (2 ticks standing)
Result: ok, 41 ticks

Decision 5
Time: tick 156; at most 344 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 24 % burns now.
You: cell (7, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 13.0 of 35.0.
Items on the floor:
  item 0 at (8, 8): worth 3.0 (started at 3); 4 cells away, about 26 ticks to walk there and pick it
  item 3 at (14, 10): worth 4.2 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 42 ticks; no fire-free route to it
  item 4 at (11, 10): worth 0.0 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 5 at (6, 9): worth 1.0 (started at 1); 2 cells away, about 18 ticks to walk there and pick it
  item 6 at (6, 4): worth 3.0 (started at 3); 8 cells away, about 42 ticks to walk there and pick it
  item 7 at (2, 4): worth 5.0 (started at 5); 14 cells away, about 66 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 7 (worth 5.0 now; about 66 ticks), after 0.2 s of thinking (2 ticks standing)
Result: ok, 66 ticks

Decision 6
Time: tick 224; at most 276 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 37 % burns now.
You: cell (3, 3), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 5.0, not banked yet. Banked so far: 13.0 of 35.0.
Items on the floor:
  item 0 at (8, 8): worth 1.6 (started at 3), BURNING: loses 0.06 per tick, worth nothing in 27 ticks; 8 cells away, about 42 ticks to walk there and pick it
  item 3 at (14, 10): worth 0.0 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 4 at (11, 10): worth 0.0 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 5 at (6, 9): worth 0.7 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 36 ticks; 14 cells away, about 66 ticks to walk there and pick it
  item 6 at (6, 4): worth 3.0 (started at 3); 2 cells away, about 18 ticks to walk there and pick it
Safe strip (rows 11 and above): 15 cells away, about 65 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 5.0; about 65 ticks), after 0.1 s of thinking (2 ticks standing)
Result: no route, 0 ticks

Decision 7
Time: tick 226; at most 274 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 38 % burns now.
You: cell (3, 3), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 5.0, not banked yet. Banked so far: 13.0 of 35.0.
Items on the floor:
  item 0 at (8, 8): worth 1.5 (started at 3), BURNING: loses 0.06 per tick, worth nothing in 25 ticks; 8 cells away, about 42 ticks to walk there and pick it
  item 3 at (14, 10): worth 0.0 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 4 at (11, 10): worth 0.0 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 5 at (6, 9): worth 0.7 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 34 ticks; no fire-free route to it
  item 6 at (6, 4): worth 3.0 (started at 3); 2 cells away, about 18 ticks to walk there and pick it
Safe strip (rows 11 and above): no fire-free route to it
Chose: get item 6 (worth 3.0 now; about 18 ticks), after 0.2 s of thinking (2 ticks standing)
Result: ok, 18 ticks

Decision 8
Time: tick 246; at most 254 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 44 % burns now.
You: cell (5, 3), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 2 item(s) worth 8.0, not banked yet. Banked so far: 13.0 of 35.0.
Items on the floor:
  item 0 at (8, 8): worth 0.3 (started at 3), BURNING: loses 0.06 per tick, worth nothing in 5 ticks; no fire-free route to it
  item 3 at (14, 10): worth 0.0 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 4 at (11, 10): worth 0.0 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 5 at (6, 9): worth 0.3 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 14 ticks; no fire-free route to it
Safe strip (rows 11 and above): no fire-free route to it
Chose: go to the safe strip, walking through fire and bank the bag (worth 8.0; no fire-free route, time unknown), after 0.1 s of thinking (2 ticks standing)
Result: episode over, 24 ticks

## Room 4: final score 0.552, ended at tick 220 with health 0; 9 decisions

Decision 1
Time: tick 0; at most 500 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 0 % burns now.
You: cell (9, 7), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 0.0 of 29.0.
Items on the floor:
  item 0 at (12, 10): worth 3.0 (started at 3); 4 cells away, about 26 ticks to walk there and pick it
  item 1 at (0, 10): worth 8.0 (started at 8); 10 cells away, about 50 ticks to walk there and pick it
  item 2 at (0, 6): worth 3.0 (started at 3); 8 cells away, about 42 ticks to walk there and pick it
  item 3 at (6, 8): worth 5.0 (started at 5); 2 cells away, about 18 ticks to walk there and pick it
  item 4 at (8, 3): worth 5.0 (started at 5); 4 cells away, about 26 ticks to walk there and pick it
  item 5 at (6, 7): worth 1.0 (started at 1); 2 cells away, about 18 ticks to walk there and pick it
  item 6 at (2, 3): worth 1.0 (started at 1); 10 cells away, about 50 ticks to walk there and pick it
  item 7 at (15, 0): worth 3.0 (started at 3); 13 cells away, about 62 ticks to walk there and pick it
Safe strip (rows 11 and above): 4 cells away, about 21 ticks to walk there and bank
Chose: get item 1 (worth 8.0 now; about 50 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 48 ticks

Decision 2
Time: tick 50; at most 450 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 2 % burns now.
You: cell (1, 9), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 8.0, not banked yet. Banked so far: 0.0 of 29.0.
Items on the floor:
  item 0 at (12, 10): worth 3.0 (started at 3); 10 cells away, about 50 ticks to walk there and pick it
  item 2 at (0, 6): worth 3.0 (started at 3); 2 cells away, about 18 ticks to walk there and pick it
  item 3 at (6, 8): worth 5.0 (started at 5); 4 cells away, about 26 ticks to walk there and pick it
  item 4 at (8, 3): worth 5.0 (started at 5); 11 cells away, about 54 ticks to walk there and pick it
  item 5 at (6, 7): worth 1.0 (started at 1); 5 cells away, about 30 ticks to walk there and pick it
  item 6 at (2, 3): worth 1.0 (started at 1); 16 cells away, about 74 ticks to walk there and pick it
  item 7 at (15, 0): worth 3.0 (started at 3); 21 cells away, about 94 ticks to walk there and pick it
Safe strip (rows 11 and above): 2 cells away, about 13 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 8.0; about 13 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 13 ticks

Decision 3
Time: tick 65; at most 435 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 2 % burns now.
You: cell (1, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 8.0 of 29.0.
Items on the floor:
  item 0 at (12, 10): worth 3.0 (started at 3); 10 cells away, about 50 ticks to walk there and pick it
  item 2 at (0, 6): worth 3.0 (started at 3); 4 cells away, about 26 ticks to walk there and pick it
  item 3 at (6, 8): worth 5.0 (started at 5); 6 cells away, about 34 ticks to walk there and pick it
  item 4 at (8, 3): worth 5.0 (started at 5); 13 cells away, about 62 ticks to walk there and pick it
  item 5 at (6, 7): worth 1.0 (started at 1); 7 cells away, about 38 ticks to walk there and pick it
  item 6 at (2, 3): worth 1.0 (started at 1); 18 cells away, about 82 ticks to walk there and pick it
  item 7 at (15, 0): worth 3.0 (started at 3); 23 cells away, about 102 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 3 (worth 5.0 now; about 34 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 34 ticks

Decision 4
Time: tick 101; at most 399 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 7 % burns now.
You: cell (5, 9), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 5.0, not banked yet. Banked so far: 8.0 of 29.0.
Items on the floor:
  item 0 at (12, 10): worth 3.0 (started at 3); 6 cells away, about 34 ticks to walk there and pick it
  item 2 at (0, 6): worth 3.0 (started at 3); 6 cells away, about 34 ticks to walk there and pick it
  item 4 at (8, 3): worth 5.0 (started at 5); no fire-free route to it
  item 5 at (6, 7): worth 1.0 (started at 1); 1 cells away, about 14 ticks to walk there and pick it
  item 6 at (2, 3): worth 1.0 (started at 1); no fire-free route to it
  item 7 at (15, 0): worth 3.0 (started at 3); no fire-free route to it
Safe strip (rows 11 and above): 2 cells away, about 13 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 5.0; about 13 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 13 ticks

Decision 5
Time: tick 116; at most 384 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 11 % burns now.
You: cell (5, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 13.0 of 29.0.
Items on the floor:
  item 0 at (12, 10): worth 3.0 (started at 3); 6 cells away, about 34 ticks to walk there and pick it
  item 2 at (0, 6): worth 3.0 (started at 3); 8 cells away, about 42 ticks to walk there and pick it
  item 4 at (8, 3): worth 5.0 (started at 5); no fire-free route to it
  item 5 at (6, 7): worth 1.0 (started at 1); 3 cells away, about 22 ticks to walk there and pick it
  item 6 at (2, 3): worth 1.0 (started at 1); no fire-free route to it
  item 7 at (15, 0): worth 3.0 (started at 3); no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 0 (worth 3.0 now; about 34 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 34 ticks

Decision 6
Time: tick 152; at most 348 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 21 % burns now.
You: cell (11, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 3.0, not banked yet. Banked so far: 13.0 of 29.0.
Items on the floor:
  item 2 at (0, 6): worth 3.0 (started at 3); 14 cells away, about 66 ticks to walk there and pick it
  item 4 at (8, 3): worth 5.0 (started at 5); no fire-free route to it
  item 5 at (6, 7): worth 0.7 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 37 ticks; 9 cells away, about 46 ticks to walk there and pick it
  item 6 at (2, 3): worth 1.0 (started at 1); no fire-free route to it
  item 7 at (15, 0): worth 3.0 (started at 3); no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 3.0; about 5 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 5 ticks

Decision 7
Time: tick 159; at most 341 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 24 % burns now.
You: cell (11, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 16.0 of 29.0.
Items on the floor:
  item 2 at (0, 6): worth 3.0 (started at 3); 14 cells away, about 66 ticks to walk there and pick it
  item 4 at (8, 3): worth 5.0 (started at 5); no fire-free route to it
  item 5 at (6, 7): worth 0.6 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 30 ticks; 9 cells away, about 46 ticks to walk there and pick it
  item 6 at (2, 3): worth 1.0 (started at 1); no fire-free route to it
  item 7 at (15, 0): worth 3.0 (started at 3); no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 2 (worth 3.0 now; about 66 ticks), after 0.3 s of thinking (4 ticks standing)
Result: no route, 0 ticks

Decision 8
Time: tick 163; at most 337 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 26 % burns now.
You: cell (11, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 16.0 of 29.0.
Items on the floor:
  item 2 at (0, 6): worth 3.0 (started at 3); no fire-free route to it
  item 4 at (8, 3): worth 5.0 (started at 5); no fire-free route to it
  item 5 at (6, 7): worth 0.5 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 26 ticks; no fire-free route to it
  item 6 at (2, 3): worth 1.0 (started at 1); no fire-free route to it
  item 7 at (15, 0): worth 3.0 (started at 3); no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 4, walking through fire (worth 5.0 now; no fire-free route, time unknown), after 0.1 s of thinking (2 ticks standing)
Result: ok, 50 ticks

Decision 9
Time: tick 215; at most 285 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 42 % burns now.
You: cell (8, 4), health 10 of 100. You are standing IN FIRE and lose 2 health per tick. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 2.1, not banked yet. Banked so far: 16.0 of 29.0.
Items on the floor:
  item 2 at (0, 6): worth 3.0 (started at 3); no fire-free route to it
  item 5 at (6, 7): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 6 at (2, 3): worth 0.7 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 35 ticks; no fire-free route to it
  item 7 at (15, 0): worth 3.0 (started at 3); no fire-free route to it
Safe strip (rows 11 and above): no fire-free route to it
Chose: go to the safe strip, walking through fire and bank the bag (worth 2.1; no fire-free route, time unknown), after 0.1 s of thinking (2 ticks standing)
Result: episode over, 3 ticks

## Room 5: final score 0.516, ended at tick 290 with health 0; 7 decisions

Decision 1
Time: tick 0; at most 500 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 0 % burns now.
You: cell (9, 7), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 0.0 of 31.0.
Items on the floor:
  item 0 at (10, 10): worth 1.0 (started at 1); 2 cells away, about 18 ticks to walk there and pick it
  item 1 at (11, 9): worth 2.0 (started at 2); 2 cells away, about 18 ticks to walk there and pick it
  item 2 at (2, 0): worth 3.0 (started at 3); 12 cells away, about 58 ticks to walk there and pick it
  item 3 at (13, 8): worth 1.0 (started at 1); 4 cells away, about 26 ticks to walk there and pick it
  item 4 at (12, 0): worth 8.0 (started at 8); 10 cells away, about 50 ticks to walk there and pick it
  item 5 at (13, 0): worth 5.0 (started at 5); 11 cells away, about 54 ticks to walk there and pick it
  item 6 at (7, 3): worth 8.0 (started at 8); 4 cells away, about 26 ticks to walk there and pick it
  item 7 at (13, 4): worth 3.0 (started at 3); 9 cells away, about 46 ticks to walk there and pick it
Safe strip (rows 11 and above): 4 cells away, about 21 ticks to walk there and bank
Chose: get item 6 (worth 8.0 now; about 26 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 24 ticks

Decision 2
Time: tick 26; at most 474 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 0 % burns now.
You: cell (8, 4), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 8.0, not banked yet. Banked so far: 0.0 of 31.0.
Items on the floor:
  item 0 at (10, 10): worth 1.0 (started at 1); 6 cells away, about 34 ticks to walk there and pick it
  item 1 at (11, 9): worth 2.0 (started at 2); 6 cells away, about 34 ticks to walk there and pick it
  item 2 at (2, 0): worth 3.0 (started at 3); 8 cells away, about 42 ticks to walk there and pick it
  item 3 at (13, 8): worth 1.0 (started at 1); 8 cells away, about 42 ticks to walk there and pick it
  item 4 at (12, 0): worth 8.0 (started at 8); 6 cells away, about 34 ticks to walk there and pick it
  item 5 at (13, 0): worth 5.0 (started at 5); 7 cells away, about 38 ticks to walk there and pick it
  item 7 at (13, 4): worth 3.0 (started at 3); 5 cells away, about 30 ticks to walk there and pick it
Safe strip (rows 11 and above): 7 cells away, about 33 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 8.0; about 33 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 33 ticks

Decision 3
Time: tick 61; at most 439 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 1 % burns now.
You: cell (8, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 8.0 of 31.0.
Items on the floor:
  item 0 at (10, 10): worth 1.0 (started at 1); 1 cells away, about 14 ticks to walk there and pick it
  item 1 at (11, 9): worth 2.0 (started at 2); 3 cells away, about 22 ticks to walk there and pick it
  item 2 at (2, 0): worth 3.0 (started at 3); 15 cells away, about 70 ticks to walk there and pick it
  item 3 at (13, 8): worth 1.0 (started at 1); 6 cells away, about 34 ticks to walk there and pick it
  item 4 at (12, 0): worth 8.0 (started at 8); 13 cells away, about 62 ticks to walk there and pick it
  item 5 at (13, 0): worth 5.0 (started at 5); 14 cells away, about 66 ticks to walk there and pick it
  item 7 at (13, 4): worth 3.0 (started at 3); 12 cells away, about 58 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 4 (worth 8.0 now; about 62 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 62 ticks

Decision 4
Time: tick 125; at most 375 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 4 % burns now.
You: cell (11, 1), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 8.0, not banked yet. Banked so far: 8.0 of 31.0.
Items on the floor:
  item 0 at (10, 10): worth 1.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 50 ticks; 13 cells away, about 62 ticks to walk there and pick it
  item 1 at (11, 9): worth 2.0 (started at 2); 17 cells away, about 78 ticks to walk there and pick it
  item 2 at (2, 0): worth 3.0 (started at 3); 8 cells away, about 42 ticks to walk there and pick it
  item 3 at (13, 8): worth 1.0 (started at 1); 19 cells away, about 86 ticks to walk there and pick it
  item 5 at (13, 0): worth 5.0 (started at 5); 1 cells away, about 14 ticks to walk there and pick it
  item 7 at (13, 4): worth 3.0 (started at 3); 3 cells away, about 22 ticks to walk there and pick it
Safe strip (rows 11 and above): 13 cells away, about 57 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 8.0; about 57 ticks), after 0.1 s of thinking (2 ticks standing)
Result: no route, 4 ticks

Decision 5
Time: tick 131; at most 369 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 5 % burns now.
You: cell (10, 1), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 8.0, not banked yet. Banked so far: 8.0 of 31.0.
Items on the floor:
  item 0 at (10, 10): worth 0.9 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 44 ticks; no fire-free route to it
  item 1 at (11, 9): worth 2.0 (started at 2); no fire-free route to it
  item 2 at (2, 0): worth 3.0 (started at 3); 7 cells away, about 38 ticks to walk there and pick it
  item 3 at (13, 8): worth 1.0 (started at 1); no fire-free route to it
  item 5 at (13, 0): worth 5.0 (started at 5); 2 cells away, about 18 ticks to walk there and pick it
  item 7 at (13, 4): worth 3.0 (started at 3); 4 cells away, about 26 ticks to walk there and pick it
Safe strip (rows 11 and above): no fire-free route to it
Chose: go to the safe strip, walking through fire and bank the bag (worth 8.0; no fire-free route, time unknown), after 0.2 s of thinking (2 ticks standing)
Result: ok, 53 ticks

Decision 6
Time: tick 186; at most 314 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 17 % burns now.
You: cell (8, 11), health 60 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 16.0 of 31.0.
Items on the floor:
  item 0 at (10, 10): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 1 at (11, 9): worth 0.8 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 21 ticks; no fire-free route to it
  item 2 at (2, 0): worth 3.0 (started at 3); no fire-free route to it
  item 3 at (13, 8): worth 1.0 (started at 1); no fire-free route to it
  item 5 at (13, 0): worth 5.0 (started at 5); no fire-free route to it
  item 7 at (13, 4): worth 3.0 (started at 3); no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 5, walking through fire (worth 5.0 now; no fire-free route, time unknown), after 0.1 s of thinking (1 ticks standing)
Result: ok, 66 ticks

Decision 7
Time: tick 253; at most 247 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 35 % burns now.
You: cell (12, 1), health 20 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 5.0, not banked yet. Banked so far: 16.0 of 31.0.
Items on the floor:
  item 0 at (10, 10): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 1 at (11, 9): worth 0.0 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 2 at (2, 0): worth 3.0 (started at 3); 9 cells away, about 46 ticks to walk there and pick it
  item 3 at (13, 8): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 7 at (13, 4): worth 3.0 (started at 3); 2 cells away, about 18 ticks to walk there and pick it
Safe strip (rows 11 and above): no fire-free route to it
Chose: go to the safe strip, walking through fire and bank the bag (worth 5.0; no fire-free route, time unknown), after 0.1 s of thinking (2 ticks standing)
Result: episode over, 35 ticks

## Room 6: final score 0.606, ended at tick 258 with health 36; 12 decisions

Decision 1
Time: tick 0; at most 500 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 0 % burns now.
You: cell (9, 7), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 0.0 of 33.0.
Items on the floor:
  item 0 at (1, 1): worth 8.0 (started at 8); 12 cells away, about 58 ticks to walk there and pick it
  item 1 at (15, 4): worth 2.0 (started at 2); 11 cells away, about 54 ticks to walk there and pick it
  item 2 at (8, 7): worth 8.0 (started at 8); 0 cells away, about 10 ticks to walk there and pick it
  item 3 at (9, 5): worth 3.0 (started at 3); 2 cells away, about 18 ticks to walk there and pick it
  item 4 at (4, 10): worth 1.0 (started at 1); 6 cells away, about 34 ticks to walk there and pick it
  item 5 at (13, 10): worth 3.0 (started at 3); 5 cells away, about 30 ticks to walk there and pick it
  item 6 at (9, 8): worth 3.0 (started at 3); 0 cells away, about 10 ticks to walk there and pick it
  item 7 at (4, 9): worth 5.0 (started at 5); 5 cells away, about 30 ticks to walk there and pick it
Safe strip (rows 11 and above): 4 cells away, about 21 ticks to walk there and bank
Chose: get item 2 (worth 8.0 now; about 10 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 10 ticks

Decision 2
Time: tick 12; at most 488 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 1 % burns now.
You: cell (9, 7), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 8.0, not banked yet. Banked so far: 0.0 of 33.0.
Items on the floor:
  item 0 at (1, 1): worth 8.0 (started at 8); 12 cells away, about 58 ticks to walk there and pick it
  item 1 at (15, 4): worth 2.0 (started at 2); 11 cells away, about 54 ticks to walk there and pick it
  item 3 at (9, 5): worth 3.0 (started at 3); 2 cells away, about 18 ticks to walk there and pick it
  item 4 at (4, 10): worth 1.0 (started at 1); 6 cells away, about 34 ticks to walk there and pick it
  item 5 at (13, 10): worth 3.0 (started at 3); 5 cells away, about 30 ticks to walk there and pick it
  item 6 at (9, 8): worth 3.0 (started at 3); 0 cells away, about 10 ticks to walk there and pick it
  item 7 at (4, 9): worth 5.0 (started at 5); 5 cells away, about 30 ticks to walk there and pick it
Safe strip (rows 11 and above): 4 cells away, about 21 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 8.0; about 21 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 21 ticks

Decision 3
Time: tick 35; at most 465 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 2 % burns now.
You: cell (9, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 8.0 of 33.0.
Items on the floor:
  item 0 at (1, 1): worth 8.0 (started at 8); 16 cells away, about 74 ticks to walk there and pick it
  item 1 at (15, 4): worth 2.0 (started at 2); 15 cells away, about 70 ticks to walk there and pick it
  item 3 at (9, 5): worth 3.0 (started at 3); 6 cells away, about 34 ticks to walk there and pick it
  item 4 at (4, 10): worth 1.0 (started at 1); 4 cells away, about 26 ticks to walk there and pick it
  item 5 at (13, 10): worth 3.0 (started at 3); 3 cells away, about 22 ticks to walk there and pick it
  item 6 at (9, 8): worth 3.0 (started at 3); 2 cells away, about 18 ticks to walk there and pick it
  item 7 at (4, 9): worth 5.0 (started at 5); 5 cells away, about 30 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 7 (worth 5.0 now; about 30 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 30 ticks

Decision 4
Time: tick 67; at most 433 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 6 % burns now.
You: cell (5, 10), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 5.0, not banked yet. Banked so far: 8.0 of 33.0.
Items on the floor:
  item 0 at (1, 1): worth 8.0 (started at 8); 15 cells away, about 70 ticks to walk there and pick it
  item 1 at (15, 4): worth 2.0 (started at 2); no fire-free route to it
  item 3 at (9, 5): worth 3.0 (started at 3); 7 cells away, about 38 ticks to walk there and pick it
  item 4 at (4, 10): worth 1.0 (started at 1); 0 cells away, about 10 ticks to walk there and pick it
  item 5 at (13, 10): worth 3.0 (started at 3); 7 cells away, about 38 ticks to walk there and pick it
  item 6 at (9, 8): worth 3.0 (started at 3); 4 cells away, about 26 ticks to walk there and pick it
Safe strip (rows 11 and above): 1 cells away, about 9 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 5.0; about 9 ticks), after 0.2 s of thinking (2 ticks standing)
Result: ok, 9 ticks

Decision 5
Time: tick 78; at most 422 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 7 % burns now.
You: cell (5, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 13.0 of 33.0.
Items on the floor:
  item 0 at (1, 1): worth 8.0 (started at 8); 16 cells away, about 74 ticks to walk there and pick it
  item 1 at (15, 4): worth 2.0 (started at 2); no fire-free route to it
  item 3 at (9, 5): worth 3.0 (started at 3); 8 cells away, about 42 ticks to walk there and pick it
  item 4 at (4, 10): worth 1.0 (started at 1); 0 cells away, about 10 ticks to walk there and pick it
  item 5 at (13, 10): worth 3.0 (started at 3); 7 cells away, about 38 ticks to walk there and pick it
  item 6 at (9, 8): worth 3.0 (started at 3); 5 cells away, about 30 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 0 (worth 8.0 now; about 74 ticks), after 0.1 s of thinking (2 ticks standing)
Result: no route, 8 ticks

Decision 6
Time: tick 88; at most 412 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 9 % burns now.
You: cell (7, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 13.0 of 33.0.
Items on the floor:
  item 0 at (1, 1): worth 8.0 (started at 8); no fire-free route to it
  item 1 at (15, 4): worth 2.0 (started at 2); no fire-free route to it
  item 3 at (9, 5): worth 3.0 (started at 3); 6 cells away, about 34 ticks to walk there and pick it
  item 4 at (4, 10): worth 1.0 (started at 1); 2 cells away, about 18 ticks to walk there and pick it
  item 5 at (13, 10): worth 3.0 (started at 3); 5 cells away, about 30 ticks to walk there and pick it
  item 6 at (9, 8): worth 3.0 (started at 3); 3 cells away, about 22 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 6 (worth 3.0 now; about 22 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 22 ticks

Decision 7
Time: tick 112; at most 388 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 11 % burns now.
You: cell (8, 9), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 3.0, not banked yet. Banked so far: 13.0 of 33.0.
Items on the floor:
  item 0 at (1, 1): worth 8.0 (started at 8); no fire-free route to it
  item 1 at (15, 4): worth 2.0 (started at 2); no fire-free route to it
  item 3 at (9, 5): worth 3.0 (started at 3); 3 cells away, about 22 ticks to walk there and pick it
  item 4 at (4, 10): worth 1.0 (started at 1); 3 cells away, about 22 ticks to walk there and pick it
  item 5 at (13, 10): worth 3.0 (started at 3); 4 cells away, about 26 ticks to walk there and pick it
Safe strip (rows 11 and above): 2 cells away, about 13 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 3.0; about 13 ticks), after 0.2 s of thinking (2 ticks standing)
Result: ok, 13 ticks

Decision 8
Time: tick 127; at most 373 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 14 % burns now.
You: cell (8, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 16.0 of 33.0.
Items on the floor:
  item 0 at (1, 1): worth 8.0 (started at 8); no fire-free route to it
  item 1 at (15, 4): worth 2.0 (started at 2); no fire-free route to it
  item 3 at (9, 5): worth 2.2 (started at 3), BURNING: loses 0.06 per tick, worth nothing in 36 ticks; 5 cells away, about 30 ticks to walk there and pick it
  item 4 at (4, 10): worth 1.0 (started at 1); 3 cells away, about 22 ticks to walk there and pick it
  item 5 at (13, 10): worth 3.0 (started at 3); 4 cells away, about 26 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 5 (worth 3.0 now; about 26 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 26 ticks

Decision 9
Time: tick 155; at most 345 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 22 % burns now.
You: cell (12, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 3.0, not banked yet. Banked so far: 16.0 of 33.0.
Items on the floor:
  item 0 at (1, 1): worth 8.0 (started at 8); no fire-free route to it
  item 1 at (15, 4): worth 2.0 (started at 2); no fire-free route to it
  item 3 at (9, 5): worth 0.5 (started at 3), BURNING: loses 0.06 per tick, worth nothing in 8 ticks; 11 cells away, about 54 ticks to walk there and pick it
  item 4 at (4, 10): worth 1.0 (started at 1); 7 cells away, about 38 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 3.0; about 5 ticks), after 0.2 s of thinking (2 ticks standing)
Result: ok, 5 ticks

Decision 10
Time: tick 162; at most 338 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 24 % burns now.
You: cell (12, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 19.0 of 33.0.
Items on the floor:
  item 0 at (1, 1): worth 8.0 (started at 8); no fire-free route to it
  item 1 at (15, 4): worth 2.0 (started at 2); no fire-free route to it
  item 3 at (9, 5): worth 0.1 (started at 3), BURNING: loses 0.06 per tick, worth nothing in 1 ticks; 11 cells away, about 54 ticks to walk there and pick it
  item 4 at (4, 10): worth 1.0 (started at 1); 7 cells away, about 38 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 4 (worth 1.0 now; about 38 ticks), after 0.2 s of thinking (3 ticks standing)
Result: ok, 38 ticks

Decision 11
Time: tick 203; at most 297 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 33 % burns now.
You: cell (5, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 1.0, not banked yet. Banked so far: 19.0 of 33.0.
Items on the floor:
  item 0 at (1, 1): worth 8.0 (started at 8); no fire-free route to it
  item 1 at (15, 4): worth 2.0 (started at 2); no fire-free route to it
  item 3 at (9, 5): worth 0.0 (started at 3), BURNING: loses 0.06 per tick, worth nothing in 0 ticks; no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 1.0; about 5 ticks), after 0.2 s of thinking (3 ticks standing)
Result: ok, 5 ticks

Decision 12
Time: tick 211; at most 289 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 35 % burns now.
You: cell (5, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 20.0 of 33.0.
Items on the floor:
  item 0 at (1, 1): worth 6.9 (started at 8), BURNING: loses 0.16 per tick, worth nothing in 43 ticks; no fire-free route to it
  item 1 at (15, 4): worth 2.0 (started at 2); no fire-free route to it
  item 3 at (9, 5): worth 0.0 (started at 3), BURNING: loses 0.06 per tick, worth nothing in 0 ticks; no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 0, walking through fire (worth 6.9 now; no fire-free route, time unknown), after 0.1 s of thinking (2 ticks standing)
Result: episode over, 45 ticks

## Room 7: final score 0.773, ended at tick 485 with health 100; 15 decisions

Decision 1
Time: tick 0; at most 500 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 0 % burns now.
You: cell (9, 7), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 0.0 of 22.0.
Items on the floor:
  item 0 at (11, 8): worth 2.0 (started at 2); 1 cells away, about 14 ticks to walk there and pick it
  item 1 at (13, 1): worth 2.0 (started at 2); 10 cells away, about 50 ticks to walk there and pick it
  item 2 at (2, 6): worth 2.0 (started at 2); 6 cells away, about 34 ticks to walk there and pick it
  item 3 at (3, 0): worth 3.0 (started at 3); 11 cells away, about 54 ticks to walk there and pick it
  item 4 at (13, 4): worth 3.0 (started at 3); 9 cells away, about 46 ticks to walk there and pick it
  item 5 at (3, 6): worth 3.0 (started at 3); 5 cells away, about 30 ticks to walk there and pick it
  item 6 at (1, 8): worth 2.0 (started at 2); 7 cells away, about 38 ticks to walk there and pick it
  item 7 at (4, 1): worth 5.0 (started at 5); 9 cells away, about 46 ticks to walk there and pick it
Safe strip (rows 11 and above): 4 cells away, about 21 ticks to walk there and bank
Chose: get item 7 (worth 5.0 now; about 46 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 44 ticks

Decision 2
Time: tick 46; at most 454 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 2 % burns now.
You: cell (5, 2), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 5.0, not banked yet. Banked so far: 0.0 of 22.0.
Items on the floor:
  item 0 at (11, 8): worth 2.0 (started at 2); 10 cells away, about 50 ticks to walk there and pick it
  item 1 at (13, 1): worth 2.0 (started at 2); 7 cells away, about 38 ticks to walk there and pick it
  item 2 at (2, 6): worth 2.0 (started at 2); 11 cells away, about 54 ticks to walk there and pick it
  item 3 at (3, 0): worth 3.0 (started at 3); 2 cells away, about 18 ticks to walk there and pick it
  item 4 at (13, 4): worth 3.0 (started at 3); 8 cells away, about 42 ticks to walk there and pick it
  item 5 at (3, 6): worth 3.0 (started at 3); 10 cells away, about 50 ticks to walk there and pick it
  item 6 at (1, 8): worth 2.0 (started at 2); 12 cells away, about 58 ticks to walk there and pick it
Safe strip (rows 11 and above): 11 cells away, about 49 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 5.0; about 49 ticks), after 0.2 s of thinking (2 ticks standing)
Result: ok, 49 ticks

Decision 3
Time: tick 97; at most 403 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 5 % burns now.
You: cell (7, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 5.0 of 22.0.
Items on the floor:
  item 0 at (11, 8): worth 2.0 (started at 2); 5 cells away, about 30 ticks to walk there and pick it
  item 1 at (13, 1): worth 0.4 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 10 ticks; 14 cells away, about 66 ticks to walk there and pick it
  item 2 at (2, 6): worth 2.0 (started at 2); 8 cells away, about 42 ticks to walk there and pick it
  item 3 at (3, 0): worth 3.0 (started at 3); 13 cells away, about 62 ticks to walk there and pick it
  item 4 at (13, 4): worth 3.0 (started at 3); 13 cells away, about 62 ticks to walk there and pick it
  item 5 at (3, 6): worth 3.0 (started at 3); 7 cells away, about 38 ticks to walk there and pick it
  item 6 at (1, 8): worth 2.0 (started at 2); 7 cells away, about 38 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 5 (worth 3.0 now; about 38 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 38 ticks

Decision 4
Time: tick 137; at most 363 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 7 % burns now.
You: cell (4, 7), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 3.0, not banked yet. Banked so far: 5.0 of 22.0.
Items on the floor:
  item 0 at (11, 8): worth 2.0 (started at 2); 6 cells away, about 34 ticks to walk there and pick it
  item 1 at (13, 1): worth 0.0 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 2 at (2, 6): worth 2.0 (started at 2); 1 cells away, about 14 ticks to walk there and pick it
  item 3 at (3, 0): worth 3.0 (started at 3); 12 cells away, about 58 ticks to walk there and pick it
  item 4 at (13, 4): worth 3.0 (started at 3); 12 cells away, about 58 ticks to walk there and pick it
  item 6 at (1, 8): worth 2.0 (started at 2); 2 cells away, about 18 ticks to walk there and pick it
Safe strip (rows 11 and above): 4 cells away, about 21 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 3.0; about 21 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 21 ticks

Decision 5
Time: tick 160; at most 340 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 12 % burns now.
You: cell (4, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 8.0 of 22.0.
Items on the floor:
  item 0 at (11, 8): worth 2.0 (started at 2); 8 cells away, about 42 ticks to walk there and pick it
  item 1 at (13, 1): worth 0.0 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 2 at (2, 6): worth 2.0 (started at 2); 5 cells away, about 30 ticks to walk there and pick it
  item 3 at (3, 0): worth 3.0 (started at 3); 16 cells away, about 74 ticks to walk there and pick it
  item 4 at (13, 4): worth 2.9 (started at 3), BURNING: loses 0.06 per tick, worth nothing in 49 ticks; no fire-free route to it
  item 6 at (1, 8): worth 2.0 (started at 2); 4 cells away, about 26 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 3 (worth 3.0 now; about 74 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 74 ticks

Decision 6
Time: tick 236; at most 264 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 17 % burns now.
You: cell (4, 1), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 3.0, not banked yet. Banked so far: 8.0 of 22.0.
Items on the floor:
  item 0 at (11, 8): worth 2.0 (started at 2); 12 cells away, about 58 ticks to walk there and pick it
  item 1 at (13, 1): worth 0.0 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 2 at (2, 6): worth 2.0 (started at 2); 13 cells away, about 62 ticks to walk there and pick it
  item 4 at (13, 4): worth 0.0 (started at 3), BURNING: loses 0.06 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 6 at (1, 8): worth 2.0 (started at 2); 14 cells away, about 66 ticks to walk there and pick it
Safe strip (rows 11 and above): 13 cells away, about 57 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 3.0; about 57 ticks), after 0.2 s of thinking (2 ticks standing)
Result: ok, 57 ticks

Decision 7
Time: tick 295; at most 205 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 19 % burns now.
You: cell (7, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 11.0 of 22.0.
Items on the floor:
  item 0 at (11, 8): worth 2.0 (started at 2); 5 cells away, about 30 ticks to walk there and pick it
  item 1 at (13, 1): worth 0.0 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 2 at (2, 6): worth 2.0 (started at 2); 8 cells away, about 42 ticks to walk there and pick it
  item 4 at (13, 4): worth 0.0 (started at 3), BURNING: loses 0.06 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 6 at (1, 8): worth 2.0 (started at 2); 7 cells away, about 38 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 0 (worth 2.0 now; about 30 ticks), after 0.2 s of thinking (2 ticks standing)
Result: ok, 30 ticks

Decision 8
Time: tick 327; at most 173 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 22 % burns now.
You: cell (10, 9), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 2.0, not banked yet. Banked so far: 11.0 of 22.0.
Items on the floor:
  item 1 at (13, 1): worth 0.0 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 2 at (2, 6): worth 2.0 (started at 2); 9 cells away, about 46 ticks to walk there and pick it
  item 4 at (13, 4): worth 0.0 (started at 3), BURNING: loses 0.06 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 6 at (1, 8): worth 2.0 (started at 2); 8 cells away, about 42 ticks to walk there and pick it
Safe strip (rows 11 and above): 2 cells away, about 13 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 2.0; about 13 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 13 ticks

Decision 9
Time: tick 342; at most 158 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 24 % burns now.
You: cell (10, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 13.0 of 22.0.
Items on the floor:
  item 1 at (13, 1): worth 0.0 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 2 at (2, 6): worth 2.0 (started at 2); 11 cells away, about 54 ticks to walk there and pick it
  item 4 at (13, 4): worth 0.0 (started at 3), BURNING: loses 0.06 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 6 at (1, 8): worth 2.0 (started at 2); 10 cells away, about 50 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 6 (worth 2.0 now; about 50 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 50 ticks

Decision 10
Time: tick 394; at most 106 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 30 % burns now.
You: cell (2, 9), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 2.0, not banked yet. Banked so far: 13.0 of 22.0.
Items on the floor:
  item 1 at (13, 1): worth 0.0 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 2 at (2, 6): worth 2.0 (started at 2); 2 cells away, about 18 ticks to walk there and pick it
  item 4 at (13, 4): worth 0.0 (started at 3), BURNING: loses 0.06 per tick, worth nothing in 0 ticks; no fire-free route to it
Safe strip (rows 11 and above): 2 cells away, about 13 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 2.0; about 13 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 13 ticks

Decision 11
Time: tick 409; at most 91 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 31 % burns now.
You: cell (2, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 15.0 of 22.0.
Items on the floor:
  item 1 at (13, 1): worth 0.0 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 2 at (2, 6): worth 2.0 (started at 2); 4 cells away, about 26 ticks to walk there and pick it
  item 4 at (13, 4): worth 0.0 (started at 3), BURNING: loses 0.06 per tick, worth nothing in 0 ticks; no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 2 (worth 2.0 now; about 26 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 26 ticks

Decision 12
Time: tick 437; at most 63 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 37 % burns now.
You: cell (2, 7), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 2.0, not banked yet. Banked so far: 15.0 of 22.0.
Items on the floor:
  item 1 at (13, 1): worth 0.0 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 4 at (13, 4): worth 0.0 (started at 3), BURNING: loses 0.06 per tick, worth nothing in 0 ticks; no fire-free route to it
Safe strip (rows 11 and above): 4 cells away, about 21 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 2.0; about 21 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 21 ticks

Decision 13
Time: tick 460; at most 40 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 40 % burns now.
You: cell (2, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 17.0 of 22.0.
Items on the floor:
  item 1 at (13, 1): worth 0.0 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 4 at (13, 4): worth 0.0 (started at 3), BURNING: loses 0.06 per tick, worth nothing in 0 ticks; no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: wait 10 ticks, after 0.1 s of thinking (2 ticks standing)
Result: ok, 10 ticks

Decision 14
Time: tick 472; at most 28 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 45 % burns now.
You: cell (2, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 17.0 of 22.0.
Items on the floor:
  item 1 at (13, 1): worth 0.0 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 4 at (13, 4): worth 0.0 (started at 3), BURNING: loses 0.06 per tick, worth nothing in 0 ticks; no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: wait 10 ticks, after 0.1 s of thinking (2 ticks standing)
Result: ok, 10 ticks

Decision 15
Time: tick 484; at most 16 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 50 % burns now.
You: cell (2, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 17.0 of 22.0.
Items on the floor:
  item 1 at (13, 1): worth 0.0 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 4 at (13, 4): worth 0.0 (started at 3), BURNING: loses 0.06 per tick, worth nothing in 0 ticks; no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: wait 10 ticks, after 0.1 s of thinking (2 ticks standing)
Result: (the episode ended first)

## Room 8: final score 0.767, ended at tick 500 with health 90; 23 decisions

Decision 1
Time: tick 0; at most 500 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 0 % burns now.
You: cell (9, 7), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 0.0 of 30.0.
Items on the floor:
  item 0 at (12, 10): worth 5.0 (started at 5); 4 cells away, about 26 ticks to walk there and pick it
  item 1 at (4, 7): worth 8.0 (started at 8); 4 cells away, about 26 ticks to walk there and pick it
  item 2 at (2, 7): worth 8.0 (started at 8); 6 cells away, about 34 ticks to walk there and pick it
  item 3 at (15, 4): worth 1.0 (started at 1); 11 cells away, about 54 ticks to walk there and pick it
  item 4 at (3, 0): worth 1.0 (started at 1); 11 cells away, about 54 ticks to walk there and pick it
  item 5 at (10, 2): worth 5.0 (started at 5); 6 cells away, about 34 ticks to walk there and pick it
  item 6 at (14, 10): worth 1.0 (started at 1); 6 cells away, about 34 ticks to walk there and pick it
  item 7 at (3, 7): worth 1.0 (started at 1); 5 cells away, about 30 ticks to walk there and pick it
Safe strip (rows 11 and above): 4 cells away, about 21 ticks to walk there and bank
Chose: get item 1 (worth 8.0 now; about 26 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 24 ticks

Decision 2
Time: tick 26; at most 474 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 1 % burns now.
You: cell (5, 7), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 8.0, not banked yet. Banked so far: 0.0 of 30.0.
Items on the floor:
  item 0 at (12, 10): worth 5.0 (started at 5); 8 cells away, about 42 ticks to walk there and pick it
  item 2 at (2, 7): worth 8.0 (started at 8); 2 cells away, about 18 ticks to walk there and pick it
  item 3 at (15, 4): worth 1.0 (started at 1); 13 cells away, about 62 ticks to walk there and pick it
  item 4 at (3, 0): worth 1.0 (started at 1); 11 cells away, about 54 ticks to walk there and pick it
  item 5 at (10, 2): worth 5.0 (started at 5); 8 cells away, about 42 ticks to walk there and pick it
  item 6 at (14, 10): worth 1.0 (started at 1); 10 cells away, about 50 ticks to walk there and pick it
  item 7 at (3, 7): worth 1.0 (started at 1); 1 cells away, about 14 ticks to walk there and pick it
Safe strip (rows 11 and above): 4 cells away, about 21 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 8.0; about 21 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 21 ticks

Decision 3
Time: tick 49; at most 451 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 2 % burns now.
You: cell (5, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 8.0 of 30.0.
Items on the floor:
  item 0 at (12, 10): worth 5.0 (started at 5); 6 cells away, about 34 ticks to walk there and pick it
  item 2 at (2, 7): worth 8.0 (started at 8); 5 cells away, about 30 ticks to walk there and pick it
  item 3 at (15, 4): worth 1.0 (started at 1); 19 cells away, about 86 ticks to walk there and pick it
  item 4 at (3, 0): worth 1.0 (started at 1); 15 cells away, about 70 ticks to walk there and pick it
  item 5 at (10, 2): worth 5.0 (started at 5); 12 cells away, about 58 ticks to walk there and pick it
  item 6 at (14, 10): worth 1.0 (started at 1); 8 cells away, about 42 ticks to walk there and pick it
  item 7 at (3, 7): worth 1.0 (started at 1); 4 cells away, about 26 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 2 (worth 8.0 now; about 30 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 30 ticks

Decision 4
Time: tick 81; at most 419 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 3 % burns now.
You: cell (3, 8), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 8.0, not banked yet. Banked so far: 8.0 of 30.0.
Items on the floor:
  item 0 at (12, 10): worth 5.0 (started at 5); 9 cells away, about 46 ticks to walk there and pick it
  item 3 at (15, 4): worth 1.0 (started at 1); 18 cells away, about 82 ticks to walk there and pick it
  item 4 at (3, 0): worth 1.0 (started at 1); 14 cells away, about 66 ticks to walk there and pick it
  item 5 at (10, 2): worth 5.0 (started at 5); 11 cells away, about 54 ticks to walk there and pick it
  item 6 at (14, 10): worth 1.0 (started at 1); 11 cells away, about 54 ticks to walk there and pick it
  item 7 at (3, 7): worth 1.0 (started at 1); 0 cells away, about 10 ticks to walk there and pick it
Safe strip (rows 11 and above): 3 cells away, about 17 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 8.0; about 17 ticks), after 0.3 s of thinking (4 ticks standing)
Result: ok, 17 ticks

Decision 5
Time: tick 102; at most 398 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 5 % burns now.
You: cell (3, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 16.0 of 30.0.
Items on the floor:
  item 0 at (12, 10): worth 5.0 (started at 5); 8 cells away, about 42 ticks to walk there and pick it
  item 3 at (15, 4): worth 1.0 (started at 1); 25 cells away, about 110 ticks to walk there and pick it
  item 4 at (3, 0): worth 1.0 (started at 1); 17 cells away, about 78 ticks to walk there and pick it
  item 5 at (10, 2): worth 5.0 (started at 5); 14 cells away, about 66 ticks to walk there and pick it
  item 6 at (14, 10): worth 1.0 (started at 1); 10 cells away, about 50 ticks to walk there and pick it
  item 7 at (3, 7): worth 1.0 (started at 1); 3 cells away, about 22 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 0 (worth 5.0 now; about 42 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 42 ticks

Decision 6
Time: tick 146; at most 354 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 8 % burns now.
You: cell (11, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 5.0, not banked yet. Banked so far: 16.0 of 30.0.
Items on the floor:
  item 3 at (15, 4): worth 0.1 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 7 ticks; no fire-free route to it
  item 4 at (3, 0): worth 1.0 (started at 1); 17 cells away, about 78 ticks to walk there and pick it
  item 5 at (10, 2): worth 5.0 (started at 5); 12 cells away, about 58 ticks to walk there and pick it
  item 6 at (14, 10): worth 1.0 (started at 1); 2 cells away, about 18 ticks to walk there and pick it
  item 7 at (3, 7): worth 1.0 (started at 1); 10 cells away, about 50 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 5.0; about 5 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 5 ticks

Decision 7
Time: tick 153; at most 347 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 9 % burns now.
You: cell (11, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 21.0 of 30.0.
Items on the floor:
  item 3 at (15, 4): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 4 at (3, 0): worth 1.0 (started at 1); 17 cells away, about 78 ticks to walk there and pick it
  item 5 at (10, 2): worth 5.0 (started at 5); 12 cells away, about 58 ticks to walk there and pick it
  item 6 at (14, 10): worth 1.0 (started at 1); 2 cells away, about 18 ticks to walk there and pick it
  item 7 at (3, 7): worth 1.0 (started at 1); 10 cells away, about 50 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 5 (worth 5.0 now; about 58 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 58 ticks

Decision 8
Time: tick 213; at most 287 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 14 % burns now.
You: cell (9, 3), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 0.0, not banked yet. Banked so far: 21.0 of 30.0.
Items on the floor:
  item 3 at (15, 4): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 4 at (3, 0): worth 1.0 (started at 1); no fire-free route to it
  item 6 at (14, 10): worth 1.0 (started at 1); no fire-free route to it
  item 7 at (3, 7): worth 1.0 (started at 1); no fire-free route to it
Safe strip (rows 11 and above): no fire-free route to it
Chose: go to the safe strip, walking through fire and bank the bag (worth 0.0; no fire-free route, time unknown), after 0.1 s of thinking (2 ticks standing)
Result: ok, 41 ticks

Decision 9
Time: tick 256; at most 244 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 17 % burns now.
You: cell (8, 11), health 90 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 21.0 of 30.0.
Items on the floor:
  item 3 at (15, 4): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 4 at (3, 0): worth 1.0 (started at 1); 14 cells away, about 66 ticks to walk there and pick it
  item 6 at (14, 10): worth 1.0 (started at 1); 5 cells away, about 30 ticks to walk there and pick it
  item 7 at (3, 7): worth 1.0 (started at 1); 7 cells away, about 38 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 6 (worth 1.0 now; about 30 ticks), after 0.2 s of thinking (2 ticks standing)
Result: ok, 30 ticks

Decision 10
Time: tick 288; at most 212 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 20 % burns now.
You: cell (13, 11), health 90 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 1.0, not banked yet. Banked so far: 21.0 of 30.0.
Items on the floor:
  item 3 at (15, 4): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 4 at (3, 0): worth 1.0 (started at 1); no fire-free route to it
  item 7 at (3, 7): worth 1.0 (started at 1); 12 cells away, about 58 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 1.0; about 5 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 5 ticks

Decision 11
Time: tick 295; at most 205 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 21 % burns now.
You: cell (13, 11), health 90 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 22.0 of 30.0.
Items on the floor:
  item 3 at (15, 4): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 4 at (3, 0): worth 1.0 (started at 1); no fire-free route to it
  item 7 at (3, 7): worth 1.0 (started at 1); 12 cells away, about 58 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 7 (worth 1.0 now; about 58 ticks), after 0.2 s of thinking (2 ticks standing)
Result: ok, 58 ticks

Decision 12
Time: tick 355; at most 145 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 23 % burns now.
You: cell (4, 8), health 90 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 1.0, not banked yet. Banked so far: 22.0 of 30.0.
Items on the floor:
  item 3 at (15, 4): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 4 at (3, 0): worth 1.0 (started at 1); no fire-free route to it
Safe strip (rows 11 and above): 3 cells away, about 17 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 1.0; about 17 ticks), after 0.2 s of thinking (2 ticks standing)
Result: ok, 17 ticks

Decision 13
Time: tick 374; at most 126 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 24 % burns now.
You: cell (4, 11), health 90 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 23.0 of 30.0.
Items on the floor:
  item 3 at (15, 4): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 4 at (3, 0): worth 1.0 (started at 1); no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: wait 10 ticks, after 0.2 s of thinking (3 ticks standing)
Result: ok, 10 ticks

Decision 14
Time: tick 387; at most 113 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 25 % burns now.
You: cell (4, 11), health 90 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 23.0 of 30.0.
Items on the floor:
  item 3 at (15, 4): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 4 at (3, 0): worth 1.0 (started at 1); no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: wait 10 ticks, after 0.1 s of thinking (1 ticks standing)
Result: ok, 10 ticks

Decision 15
Time: tick 398; at most 102 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 25 % burns now.
You: cell (4, 11), health 90 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 23.0 of 30.0.
Items on the floor:
  item 3 at (15, 4): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 4 at (3, 0): worth 1.0 (started at 1); no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: wait 10 ticks, after 0.2 s of thinking (2 ticks standing)
Result: ok, 10 ticks

Decision 16
Time: tick 410; at most 90 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 26 % burns now.
You: cell (4, 11), health 90 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 23.0 of 30.0.
Items on the floor:
  item 3 at (15, 4): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 4 at (3, 0): worth 1.0 (started at 1); no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: wait 10 ticks, after 0.1 s of thinking (2 ticks standing)
Result: ok, 10 ticks

Decision 17
Time: tick 422; at most 78 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 27 % burns now.
You: cell (4, 11), health 90 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 23.0 of 30.0.
Items on the floor:
  item 3 at (15, 4): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 4 at (3, 0): worth 1.0 (started at 1); no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: wait 10 ticks, after 0.2 s of thinking (2 ticks standing)
Result: ok, 10 ticks

Decision 18
Time: tick 434; at most 66 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 29 % burns now.
You: cell (4, 11), health 90 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 23.0 of 30.0.
Items on the floor:
  item 3 at (15, 4): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 4 at (3, 0): worth 1.0 (started at 1); no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: wait 10 ticks, after 0.1 s of thinking (2 ticks standing)
Result: ok, 10 ticks

Decision 19
Time: tick 446; at most 54 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 31 % burns now.
You: cell (4, 11), health 90 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 23.0 of 30.0.
Items on the floor:
  item 3 at (15, 4): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 4 at (3, 0): worth 1.0 (started at 1); no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: wait 10 ticks, after 0.2 s of thinking (2 ticks standing)
Result: ok, 10 ticks

Decision 20
Time: tick 458; at most 42 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 32 % burns now.
You: cell (4, 11), health 90 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 23.0 of 30.0.
Items on the floor:
  item 3 at (15, 4): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 4 at (3, 0): worth 0.8 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 40 ticks; no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: wait 10 ticks, after 0.2 s of thinking (2 ticks standing)
Result: ok, 10 ticks

Decision 21
Time: tick 470; at most 30 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 33 % burns now.
You: cell (4, 11), health 90 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 23.0 of 30.0.
Items on the floor:
  item 3 at (15, 4): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 4 at (3, 0): worth 0.6 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 28 ticks; no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: wait 10 ticks, after 0.1 s of thinking (2 ticks standing)
Result: ok, 10 ticks

Decision 22
Time: tick 482; at most 18 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 34 % burns now.
You: cell (4, 11), health 90 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 23.0 of 30.0.
Items on the floor:
  item 3 at (15, 4): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 4 at (3, 0): worth 0.3 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 16 ticks; no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: wait 10 ticks, after 0.1 s of thinking (2 ticks standing)
Result: ok, 10 ticks

Decision 23
Time: tick 494; at most 6 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 35 % burns now.
You: cell (4, 11), health 90 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 23.0 of 30.0.
Items on the floor:
  item 3 at (15, 4): worth 0.0 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 4 at (3, 0): worth 0.1 (started at 1), BURNING: loses 0.02 per tick, worth nothing in 4 ticks; no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: wait 10 ticks, after 0.2 s of thinking (2 ticks standing)
Result: episode over, 4 ticks

## Room 9: final score 0.618, ended at tick 226 with health 70; 11 decisions

Decision 1
Time: tick 0; at most 500 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 0 % burns now.
You: cell (9, 7), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 0.0 of 25.0.
Items on the floor:
  item 0 at (12, 8): worth 2.0 (started at 2); 2 cells away, about 18 ticks to walk there and pick it
  item 1 at (3, 10): worth 3.0 (started at 3); 7 cells away, about 38 ticks to walk there and pick it
  item 2 at (5, 8): worth 2.0 (started at 2); 3 cells away, about 22 ticks to walk there and pick it
  item 3 at (1, 4): worth 5.0 (started at 5); 11 cells away, about 54 ticks to walk there and pick it
  item 4 at (14, 8): worth 3.0 (started at 3); 4 cells away, about 26 ticks to walk there and pick it
  item 5 at (3, 9): worth 5.0 (started at 5); 6 cells away, about 34 ticks to walk there and pick it
  item 6 at (5, 9): worth 3.0 (started at 3); 5 cells away, about 30 ticks to walk there and pick it
  item 7 at (6, 5): worth 2.0 (started at 2); 3 cells away, about 22 ticks to walk there and pick it
Safe strip (rows 11 and above): 4 cells away, about 21 ticks to walk there and bank
Chose: get item 5 (worth 5.0 now; about 34 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 32 ticks

Decision 2
Time: tick 34; at most 466 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 2 % burns now.
You: cell (4, 8), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 5.0, not banked yet. Banked so far: 0.0 of 25.0.
Items on the floor:
  item 0 at (12, 8): worth 2.0 (started at 2); 8 cells away, about 42 ticks to walk there and pick it
  item 1 at (3, 10): worth 3.0 (started at 3); 1 cells away, about 14 ticks to walk there and pick it
  item 2 at (5, 8): worth 1.1 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 27 ticks; 0 cells away, about 10 ticks to walk there and pick it
  item 3 at (1, 4): worth 5.0 (started at 5); 17 cells away, about 78 ticks to walk there and pick it
  item 4 at (14, 8): worth 3.0 (started at 3); 10 cells away, about 50 ticks to walk there and pick it
  item 6 at (5, 9): worth 3.0 (started at 3); 0 cells away, about 10 ticks to walk there and pick it
  item 7 at (6, 5): worth 2.0 (started at 2); 9 cells away, about 46 ticks to walk there and pick it
Safe strip (rows 11 and above): 3 cells away, about 17 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 5.0; about 17 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 17 ticks

Decision 3
Time: tick 53; at most 447 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 4 % burns now.
You: cell (4, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 5.0 of 25.0.
Items on the floor:
  item 0 at (12, 8): worth 2.0 (started at 2); 9 cells away, about 46 ticks to walk there and pick it
  item 1 at (3, 10): worth 3.0 (started at 3); 0 cells away, about 10 ticks to walk there and pick it
  item 2 at (5, 8): worth 0.3 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 8 ticks; 2 cells away, about 18 ticks to walk there and pick it
  item 3 at (1, 4): worth 5.0 (started at 5); 20 cells away, about 90 ticks to walk there and pick it
  item 4 at (14, 8): worth 3.0 (started at 3); 11 cells away, about 54 ticks to walk there and pick it
  item 6 at (5, 9): worth 3.0 (started at 3); 1 cells away, about 14 ticks to walk there and pick it
  item 7 at (6, 5): worth 2.0 (started at 2); 12 cells away, about 58 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 1 (worth 3.0 now; about 10 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 10 ticks

Decision 4
Time: tick 65; at most 435 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 6 % burns now.
You: cell (4, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 3.0, not banked yet. Banked so far: 5.0 of 25.0.
Items on the floor:
  item 0 at (12, 8): worth 2.0 (started at 2); 9 cells away, about 46 ticks to walk there and pick it
  item 2 at (5, 8): worth 0.0 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 0 ticks; 2 cells away, about 18 ticks to walk there and pick it
  item 3 at (1, 4): worth 5.0 (started at 5); 20 cells away, about 90 ticks to walk there and pick it
  item 4 at (14, 8): worth 3.0 (started at 3); 11 cells away, about 54 ticks to walk there and pick it
  item 6 at (5, 9): worth 3.0 (started at 3); 1 cells away, about 14 ticks to walk there and pick it
  item 7 at (6, 5): worth 2.0 (started at 2); 12 cells away, about 58 ticks to walk there and pick it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 3.0; about 5 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 5 ticks

Decision 5
Time: tick 72; at most 428 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 8 % burns now.
You: cell (4, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 8.0 of 25.0.
Items on the floor:
  item 0 at (12, 8): worth 2.0 (started at 2); 9 cells away, about 46 ticks to walk there and pick it
  item 2 at (5, 8): worth 0.0 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 0 ticks; 2 cells away, about 18 ticks to walk there and pick it
  item 3 at (1, 4): worth 5.0 (started at 5); no fire-free route to it
  item 4 at (14, 8): worth 3.0 (started at 3); 11 cells away, about 54 ticks to walk there and pick it
  item 6 at (5, 9): worth 3.0 (started at 3); 1 cells away, about 14 ticks to walk there and pick it
  item 7 at (6, 5): worth 2.0 (started at 2); no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 6 (worth 3.0 now; about 14 ticks), after 0.2 s of thinking (2 ticks standing)
Result: ok, 14 ticks

Decision 6
Time: tick 88; at most 412 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 9 % burns now.
You: cell (4, 10), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 2.5, not banked yet. Banked so far: 8.0 of 25.0.
Items on the floor:
  item 0 at (12, 8): worth 2.0 (started at 2); 10 cells away, about 50 ticks to walk there and pick it
  item 2 at (5, 8): worth 0.0 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 0 ticks; 1 cells away, about 14 ticks to walk there and pick it
  item 3 at (1, 4): worth 5.0 (started at 5); no fire-free route to it
  item 4 at (14, 8): worth 3.0 (started at 3); 12 cells away, about 58 ticks to walk there and pick it
  item 7 at (6, 5): worth 2.0 (started at 2); no fire-free route to it
Safe strip (rows 11 and above): 1 cells away, about 9 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 2.5; about 9 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 9 ticks

Decision 7
Time: tick 99; at most 401 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 12 % burns now.
You: cell (4, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 10.5 of 25.0.
Items on the floor:
  item 0 at (12, 8): worth 2.0 (started at 2); 9 cells away, about 46 ticks to walk there and pick it
  item 2 at (5, 8): worth 0.0 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 3 at (1, 4): worth 5.0 (started at 5); no fire-free route to it
  item 4 at (14, 8): worth 3.0 (started at 3); 11 cells away, about 54 ticks to walk there and pick it
  item 7 at (6, 5): worth 2.0 (started at 2); no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 4 (worth 3.0 now; about 54 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 54 ticks

Decision 8
Time: tick 155; at most 345 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 26 % burns now.
You: cell (13, 9), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 3.0, not banked yet. Banked so far: 10.5 of 25.0.
Items on the floor:
  item 0 at (12, 8): worth 2.0 (started at 2); 0 cells away, about 10 ticks to walk there and pick it
  item 2 at (5, 8): worth 0.0 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 3 at (1, 4): worth 5.0 (started at 5); no fire-free route to it
  item 7 at (6, 5): worth 1.3 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 33 ticks; no fire-free route to it
Safe strip (rows 11 and above): 2 cells away, about 13 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 3.0; about 13 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 13 ticks

Decision 9
Time: tick 170; at most 330 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 30 % burns now.
You: cell (13, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 13.5 of 25.0.
Items on the floor:
  item 0 at (12, 8): worth 2.0 (started at 2); 2 cells away, about 18 ticks to walk there and pick it
  item 2 at (5, 8): worth 0.0 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 3 at (1, 4): worth 5.0 (started at 5); no fire-free route to it
  item 7 at (6, 5): worth 0.7 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 18 ticks; no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 0 (worth 2.0 now; about 18 ticks), after 0.1 s of thinking (2 ticks standing)
Result: ok, 18 ticks

Decision 10
Time: tick 190; at most 310 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 35 % burns now.
You: cell (13, 9), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: 1 item(s) worth 2.0, not banked yet. Banked so far: 13.5 of 25.0.
Items on the floor:
  item 2 at (5, 8): worth 0.0 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 3 at (1, 4): worth 5.0 (started at 5); no fire-free route to it
  item 7 at (6, 5): worth 0.0 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 0 ticks; no fire-free route to it
Safe strip (rows 11 and above): 2 cells away, about 13 ticks to walk there and bank
Chose: go to the safe strip and bank the bag (worth 2.0; about 13 ticks), after 0.2 s of thinking (2 ticks standing)
Result: ok, 13 ticks

Decision 11
Time: tick 205; at most 295 ticks are left (one tick is 0.1 s). The episode also ends when 50 % of the floor burns; 43 % burns now.
You: cell (13, 11), health 100 of 100. At health 0 the episode ends and the bag is lost.
Bag: empty. Banked so far: 15.5 of 25.0.
Items on the floor:
  item 2 at (5, 8): worth 0.0 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 0 ticks; no fire-free route to it
  item 3 at (1, 4): worth 5.0 (started at 5); no fire-free route to it
  item 7 at (6, 5): worth 0.0 (started at 2), BURNING: loses 0.04 per tick, worth nothing in 0 ticks; no fire-free route to it
Safe strip (rows 11 and above): 0 cells away, about 5 ticks to walk there and bank
Chose: get item 3, walking through fire (worth 5.0 now; no fire-free route, time unknown), after 0.1 s of thinking (1 ticks standing)
Result: episode over, 20 ticks

# Task

Write the next program. Keep what worked and fix what the recorded outcomes show going wrong (value lost to fire, a bag lost because the way home closed or the episode ended, health lost, time spent standing or walking for little). Return the whole module in one ```python block.