"""Fast numpy game core replicating santorini.game.Game move-for-move.

Only the 2-player / 2-workers-per-player configuration is supported. Parity
with the reference engine is enforced by lockstep fuzz tests in
tests/test_az_fastgame.py; rule subtleties intentionally replicated:

- A move onto height 3 wins immediately and no build is applied, but the
  action's build part must still point at *any adjacent on-board cell*
  (even occupied or capped), mirroring Board._valid_move_then_build_positions.
- Building on the square the worker just vacated is legal; the capped check
  runs before the own-square exemption, exactly as in Board.
- If the player to move has no legal action the previous player wins.

Conventions specific to this module:

- `heights` is indexed [x][y]; action encoding is y-major
  (space index = y*5 + x), matching santorini.utils.
- `step` always flips `player`, including on a winning move (Game leaves
  `current_player_idx` on the winner there). Hence at any terminal state the
  player to move is the loser and `terminal_value` is -1. Cross-validation
  compares winner identity, not whose "turn" the terminal state shows.
- `step` assumes the action is legal (MCTS/self-play only feed legal ids).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from santorini import utils
from santorini.config import GRID_SIZE, MAX_BUILDING_HEIGHT
from santorini.game import Game, GameState

N = GRID_SIZE  # 5
WIN_H = MAX_BUILDING_HEIGHT  # 3
DOME = MAX_BUILDING_HEIGHT + 1  # 4
NUM_ACTIONS = N * N * 8 * 8  # 1600
NUM_SPACES = N * N  # 25; placement actions use indices 0..24
TOTAL_WORKERS = 4
OBS_CHANNELS = 13
DIRS = tuple(utils.DIRS)


@dataclass
class State:
    heights: np.ndarray  # (5,5) int8, [x][y]; 4 = dome
    workers: np.ndarray  # (2,2,2) int8, [player][worker_id][(x,y)]; -1 = unplaced
    placed: int  # workers placed so far; SETUP iff < 4
    player: int  # player to move
    winner: int  # -1 while ongoing, else winning player id

    def copy(self) -> "State":
        return State(
            self.heights.copy(), self.workers.copy(), self.placed, self.player, self.winner
        )


def initial_state() -> State:
    return State(
        heights=np.zeros((N, N), dtype=np.int8),
        workers=np.full((2, 2, 2), -1, dtype=np.int8),
        placed=0,
        player=0,
        winner=-1,
    )


def is_setup(state: State) -> bool:
    return state.placed < TOTAL_WORKERS


def is_terminal(state: State) -> bool:
    return state.winner != -1


def terminal_value(state: State) -> float:
    """Value of a terminal state from the perspective of the player to move."""
    return 1.0 if state.winner == state.player else -1.0


def _occupied(state: State) -> set[tuple[int, int]]:
    occ = set()
    for px, py in state.workers.reshape(4, 2).tolist():
        if px >= 0:
            occ.add((px, py))
    return occ


def legal_actions(state: State) -> list[int]:
    """Sorted list of legal action ids for the player to move."""
    if state.winner != -1:
        return []
    occ = _occupied(state)
    if state.placed < TOTAL_WORKERS:
        return [y * N + x for y in range(N) for x in range(N) if (x, y) not in occ]

    h = state.heights.tolist()
    actions: list[int] = []
    for wx, wy in state.workers[state.player].tolist():
        from_h = h[wx][wy]
        for mdir, (dx, dy) in enumerate(DIRS):
            tx, ty = wx + dx, wy + dy
            if not (0 <= tx < N and 0 <= ty < N):
                continue
            if (tx, ty) in occ:
                continue
            th = h[tx][ty]
            if th > from_h + 1:
                continue
            base = (wy * N + wx) * 64 + mdir * 8
            if th == WIN_H:
                # Winning move: build part may be any adjacent on-board cell.
                for bdir, (bdx, bdy) in enumerate(DIRS):
                    if 0 <= tx + bdx < N and 0 <= ty + bdy < N:
                        actions.append(base + bdir)
            else:
                for bdir, (bdx, bdy) in enumerate(DIRS):
                    bx, by = tx + bdx, ty + bdy
                    if not (0 <= bx < N and 0 <= by < N):
                        continue
                    if h[bx][by] == DOME:
                        continue
                    if (bx, by) in occ and (bx, by) != (wx, wy):
                        continue
                    actions.append(base + bdir)
    actions.sort()
    return actions


def legal_mask(state: State) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    acts = legal_actions(state)
    if acts:
        mask[acts] = True
    return mask


def _has_any_move(state: State) -> bool:
    """Early-exit existence check equivalent to `bool(legal_actions(state))`."""
    if state.placed < TOTAL_WORKERS:
        return True  # 25 cells, at most 4 workers
    h = state.heights.tolist()
    occ = _occupied(state)
    for wx, wy in state.workers[state.player].tolist():
        from_h = h[wx][wy]
        for dx, dy in DIRS:
            tx, ty = wx + dx, wy + dy
            if not (0 <= tx < N and 0 <= ty < N):
                continue
            if (tx, ty) in occ:
                continue
            th = h[tx][ty]
            if th > from_h + 1:
                continue
            if th == WIN_H:
                return True  # every cell has an adjacent on-board cell
            for bdx, bdy in DIRS:
                bx, by = tx + bdx, ty + bdy
                if not (0 <= bx < N and 0 <= by < N):
                    continue
                if h[bx][by] == DOME:
                    continue
                if (bx, by) in occ and (bx, by) != (wx, wy):
                    continue
                return True
    return False


def step(state: State, action: int) -> State:
    if state.winner != -1:
        raise ValueError("Cannot step a terminal state")
    heights = state.heights.copy()
    workers = state.workers.copy()

    if state.placed < TOTAL_WORKERS:
        x, y = action % N, action // N
        workers[state.player, state.placed // 2] = (x, y)
        placed = state.placed + 1
        nxt = 0 if placed == TOTAL_WORKERS else placed % 2
        new = State(heights, workers, placed, nxt, -1)
        if placed == TOTAL_WORKERS and not _has_any_move(new):
            new.winner = 1 - nxt
        return new

    from_idx, rem = divmod(action, 64)
    mdir, bdir = divmod(rem, 8)
    fx, fy = from_idx % N, from_idx // N
    dx, dy = DIRS[mdir]
    tx, ty = fx + dx, fy + dy

    p = state.player
    if workers[p, 0, 0] == fx and workers[p, 0, 1] == fy:
        wi = 0
    elif workers[p, 1, 0] == fx and workers[p, 1, 1] == fy:
        wi = 1
    else:
        raise ValueError(f"No worker of player {p} at {(fx, fy)}")
    workers[p, wi] = (tx, ty)

    if heights[tx, ty] == WIN_H:
        return State(heights, workers, state.placed, 1 - p, p)

    bdx, bdy = DIRS[bdir]
    heights[tx + bdx, ty + bdy] += 1
    new = State(heights, workers, state.placed, 1 - p, -1)
    if not _has_any_move(new):
        new.winner = p
    return new


def key(state: State) -> bytes:
    """Hashable id, canonical to the player to move (own workers first)."""
    p = state.player
    return (
        state.heights.tobytes()
        + state.workers[p].tobytes()
        + state.workers[1 - p].tobytes()
        + bytes((state.placed,))
    )


def canonical_obs(state: State) -> np.ndarray:
    """(5,5,13) float32 observation from the player to move's perspective.

    ch0-10 match Board.get_observation(state.player) exactly; ch11/12 are
    threat planes: empty level-3 cells adjacent to a (mine / opponent's)
    worker standing at level >= 2 — i.e. cells where that side has a legal
    winning step. (GreedyOpponent._winnable_cells ignores occupancy; this is
    the exact-legality variant, an intentional difference.)
    """
    obs = np.zeros((N, N, OBS_CHANNELS), dtype=np.float32)
    for lvl in range(DOME + 1):
        obs[:, :, lvl] = state.heights == lvl

    p = state.player
    h = state.heights.tolist()
    occ = _occupied(state)
    for rel, q in enumerate((p, 1 - p)):
        for i in range(2):
            x, y = state.workers[q, i].tolist()
            if x < 0:
                continue
            obs[x, y, 5 + 2 * rel + i] = 1.0
            obs[x, y, 9 + rel] = 1.0
            if h[x][y] >= WIN_H - 1:
                for dx, dy in DIRS:
                    tx, ty = x + dx, y + dy
                    if (
                        0 <= tx < N
                        and 0 <= ty < N
                        and h[tx][ty] == WIN_H
                        and (tx, ty) not in occ
                    ):
                        obs[tx, ty, 11 + rel] = 1.0
    return obs


def from_game(game: Game) -> State:
    """Convert a reference Game (past PLAYER_SELECT) to a fast state.

    At terminal states `player` keeps Game's current_player_idx, which for
    height-wins is the winner (Game does not advance the turn there); rely on
    `winner`, not `player`, when interpreting terminal states from this path.
    """
    if game.state == GameState.PLAYER_SELECT:
        raise ValueError("Game has not selected players yet")
    if len(game.players) != 2:
        raise ValueError("Only 2-player games are supported")

    heights = np.zeros((N, N), dtype=np.int8)
    for x in range(N):
        for y in range(N):
            heights[x, y] = game.board.get_height((x, y))
    workers = np.full((2, 2, 2), -1, dtype=np.int8)
    placed = 0
    for p, pl in enumerate(game.players):
        for w in pl.workers:
            workers[p, w.get_id()] = w.position
            placed += 1
    winner = game.winner.get_id() if game.winner is not None else -1
    return State(heights, workers, placed, game.current_player_idx, winner)


def transform_state(state: State, t: int) -> State:
    """Apply D4 transform `t` to a state (tests only)."""
    from santorini.az import symmetry

    heights = np.empty_like(state.heights)
    for x in range(N):
        for y in range(N):
            tx, ty = symmetry.apply_xy(t, x, y)
            heights[tx, ty] = state.heights[x, y]
    workers = state.workers.copy()
    for p in range(2):
        for i in range(2):
            x, y = state.workers[p, i].tolist()
            if x >= 0:
                workers[p, i] = symmetry.apply_xy(t, x, y)
    return State(heights, workers, state.placed, state.player, state.winner)
