# Santorini Multi-Agent Reinforcement Learning

## Summary

- This repo contains a custom implementation of the game Santorini, along with a custom PettingZoo/Gymnasium environment for multi-agent reinforcement learning using Stable Baseline 3.

- Supports playing against an AI with a GUI.

- [Santorini](https://en.wikipedia.org/wiki/Santorini_(game)), is an abstract board game, first [published in 2004](https://boardgamegeek.com/boardgame/9963/santorini). It is a two to three player game that is typically played on a $5 \times 5$ grid.

![Playing Santorini](/images/santorini.png)

## 📜 Rules
The [rules](http://www.boardspace.net/santorini/english/santorini-rules.html) are fairly simple. Each player sets up the game by placing two workers on the board. Each turn, a worker can be moved to an adjacent square, followed by building on a square adjacent to the square that worker just moved to.

Building increases the height of that square by one. The move destination space must be unoccupied by a worker. The height must be no more than one level higher than your starting height. If a worker builds on a space with height three, that space is capped and can no longer be moved to or built on.

When a player's piece reaches the third level, that player wins. If a player cannot move any of their pieces, that player loses.

## 🔧 Installation

### Clone the Repository:
```
git clone https://github.com/KennethJAllen/santorini
cd santorini
```
### Create Virtual Environment

- Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if not already installed.
- Sync environments
    - `uv sync`

## ♟️ Playing the Game

### GUI

`uv run santoini`

### CLI

`uv run cli`

### Browser

To play in the browser, visit https://www.kennethallenmath.com/santorini/.

Note: This is an old version of the game and currently does not support playing against an AI.

## 🏋️ Training

Train a MaskablePPO model with frozen-opponent self-play (see `uv run train --help` for flags):

```
uv run train
```

The learner plays a single seat (randomized per episode) against a pool of frozen snapshots of itself (plus a random opponent for variety); the pool is refreshed every `--snapshot-freq` steps. Rewards are pure win/loss (+1/−1); optional potential-based height shaping is available via `--shaping-scale`.

Watch metrics live in another terminal:

```
uv run tensorboard --logdir models/tb
```

Key signals:

- `eval/winrate_vs_greedy_p0` / `_p1` — winrate against a 1-ply heuristic from each seat; the main convergence signal. Winrate vs random saturates early and cannot distinguish a strong policy from a degenerate one.
- `eval/traj_diversity` — fraction of distinct trajectories across eval games. Near 0 means self-play has collapsed into a single line of play.
- `selfplay/learner_winrate` — should hover near 0.5 once the pool holds recent snapshots; ~0 or ~1 per seat is a red flag.
- `selfplay/terminal_reward_mean` — should sit strictly between −1 and 1 (both wins and losses are being experienced).

Analyze runs for convergence/collapse after (or during) training:

```
uv run python -m santorini.analyze_training run models/tb
```

## 🤖 PettingZoo Environment

The Santorini Env is set up for multi-agent reinforcement learning via custom PettingZoo environment and Stable Baseline 3.

### Action Space

- An action is a discrete integer from $0$ to $1600 = 5 \times 5 \times 8 \times 8$.

- The first $5 \times 5$ represents the board space.

- The first $8$ represents a compass direction to move (NW, N, NE, E, SE, S, SW, W).

- The second $8$ represents a compass direction to build.

### Observation Space

The observation space is a $5 \times 5$ grid representing the board with 7 channels:

- channel 0-4: building heights.
    - 0: 1 nothing if no building, else 0
    - 1: 1 for height 1, else 0
    - 2: 1 for height 2, else 0
    - 3: 1 for height 3, else 0
    - 4: 1 if capped, else 0
- channel 5-6: which player's piece occupies each cell.
    - 5: 1 if the turn player occupies the cell, else 0
    - 6: 1 if the other player occupies the cell, else 0

#### Symmetries
Santorini has 8-fold dihedral symmetry, meaning it can be rotated in 4 directions or flipped in 4 directions and the board state would be equivalent.

## Credits

- Python WebAssembly by Pygbag.

- Assets were created with ChatGPT and Aseprite.

- Stable Baseline 3 traning script based on script by Elliot from Farma Foundation
