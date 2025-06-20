# Santorini

[Santorini](https://en.wikipedia.org/wiki/Santorini_(game)), is an abstract board game, first [published in 2004](https://boardgamegeek.com/boardgame/9963/santorini). It is a two to three player game that is typically played on a $5 \times 5$ grid.


![Santorini being played](/images/santorini.png)

## 📜 Rules
The [rules](http://www.boardspace.net/santorini/english/santorini-rules.html) are fairly simple. Each player sets up the game by placing two workers on the board. Each turn, a worker can be moved to an adjacent square, followed by building on a square adjacent to the square that worker just moved to.

Building increases the height of that square by one. The move destination space must be unoccupied by a worker. The height must be no more than one level higher than your starting height. If a worker builds on a space with height three, that space is capped and can no longer be moved to or built on.

When a player's piece reaches the third level, that player wins. If a player cannot move any of their pieces, that player loses.

## ♟️ Playing the Game

### In the browser

To play in the browser, visit https://www.kennethallenmath.com/santorini/.

### Using CLI

`uv run cli`

## 🔧 Installation

### Clone the Repository:

git clone https://github.com/KennethJAllen/santorini
cd santorini
```
### Create Virtual Environment

- Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if not already installed.
- Sync environments
    - `uv sync`

## Santorini Env

The Santorini Env is set up for multi-agent reinforcement learning via PettingZoo.

### Action Space

An action is a discrete integer from $0$ to $1600 = 5 \times 5 \times 8 \times 8$.

### Observation Space

The observation space is a $5 \times 5$ grid representing the board with 6 channels:

- Channel 0:
    - Position height
- Channel 1:
    - 1 for first player piece
    - 2 for second player piece
    - 0 for no players
- Channel 2:
    - All 0 for first player's turn
    - All 1 for second player's turn

#### Symmetries
Santorini has 8-fold dihedral symmetry, meaning it can be rotated in 4 directions or flipped in 4 directions and the board state would be equivalent.

## Credits

Python WebAssembly by Pygbag.

Assets were created with Aseprite.
