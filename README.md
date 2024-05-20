# santorini
[Santorini](https://boardgamegeek.com/boardgame/194655/santorini), is an abstract board game, first [published in 2004](https://boardgamegeek.com/boardgame/9963/santorini). It is a two to thre player game that is typically played on a $5 \times 5$ grid.

## 📜 Rules
The [rules](http://www.boardspace.net/santorini/english/santorini-rules.html) are fairly simple. Each player sets up the game by placing two workers on the board. Each turn, a worker can be moved to an adjacent square, followed by building on a square adjacent to the square that worker just moved to.

Building increases the height of that square by one. The move destination space must be unoccupied by a worker. The height must be no more than one level higher than your starting height. If a worker builds on a space with height three, that space is capped and can no longer be moved to or built on.

When a player's piece reaches the third level, that player wins. If a player cannot move any of their pieces, that player loses.

## 🔧 Installation

### Clone the Repository:

```
git clone https://github.com/KennethJAllen/YGO-small-world
cd YGO-small-world
```
### Install Dependencies with Poetry:

*   Install Poetry if not already installed.
*   Run the following command in the project directory:

```
poetry install
```
### Activate the Virtual Environment:
```
poetry shell
```
You can now run the project's scripts within the poetry shell.

## ♟️ Playing the Game

### Using `main.py`

To play the game, execute the script `main.py`.

On each turn, a player will be prompted to choose a square to move from, choose a square to move to, and place a piece.

The squares on the board are specified by [algebraic notation](https://en.wikipedia.org/wiki/Algebraic_notation_(chess)), meaning each square is identified by a letter-coordinate pair. On a typical $5 \times 5$ board, the squares are labeled `A1` through `E5`.
