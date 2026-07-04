"""Opponent policies for frozen-opponent self-play training and evaluation.

All opponents implement `choose_action(obs, mask) -> int` where `obs` is the
(5, 5, 11) board observation from the opponent's own perspective (see
`Board.get_observation`) and `mask` is the flat 1600-dim valid-action mask.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from santorini import utils
from santorini.config import GRID_SIZE, MAX_BUILDING_HEIGHT, NUM_WORKERS


class Opponent:
    """Protocol for opponent policies."""

    def choose_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        raise NotImplementedError


class RandomOpponent(Opponent):
    """Picks a uniformly random valid action."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng if rng is not None else np.random.default_rng()

    def choose_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        valid = np.flatnonzero(mask)
        return int(self.rng.choice(valid))


class GreedyOpponent(Opponent):
    """1-ply heuristic baseline: win if possible, block an imminent opponent
    win by capping the threatened cell, avoid gifting a reachable level-3
    square, otherwise prefer climbing.

    Unlike a random opponent this punishes obvious blunders, so winrate
    against it does not saturate for degenerate policies.
    """

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng if rng is not None else np.random.default_rng()

    def choose_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        valid = np.flatnonzero(mask)
        heights = np.argmax(obs[:, :, : MAX_BUILDING_HEIGHT + 2], axis=2)
        my_cells = obs[:, :, 9]
        opp_cells = obs[:, :, 10]

        # Setup phase: fewer than 2*NUM_WORKERS workers on the board means we
        # are still placing. Prefer central squares.
        if my_cells.sum() + opp_cells.sum() < 2 * NUM_WORKERS:
            return self._choose_placement(valid)

        threatened = self._winnable_cells(heights, opp_cells)

        best_score = -np.inf
        best_actions: list[int] = []
        for action in valid:
            move_from, move_to, build_on = utils.decode_action(int(action))
            score = self._score_action(heights, move_from, move_to, build_on, threatened)
            if score > best_score:
                best_score = score
                best_actions = [int(action)]
            elif score == best_score:
                best_actions.append(int(action))
        return int(self.rng.choice(best_actions))

    def _choose_placement(self, valid: np.ndarray) -> int:
        center = (GRID_SIZE - 1) / 2
        dists = []
        for action in valid:
            x, y = utils.decode_space(int(action))
            dists.append(abs(x - center) + abs(y - center))
        dists = np.array(dists)
        best = valid[dists == dists.min()]
        return int(self.rng.choice(best))

    def _winnable_cells(self, heights: np.ndarray, opp_cells: np.ndarray) -> set[tuple[int, int]]:
        """Level-3 cells an opposing worker (standing at level >= 2) can step onto."""
        cells: set[tuple[int, int]] = set()
        for x, y in zip(*np.nonzero(opp_cells)):
            if heights[x, y] < MAX_BUILDING_HEIGHT - 1:
                continue
            for dx, dy in utils.DIRS:
                tx, ty = x + dx, y + dy
                if 0 <= tx < GRID_SIZE and 0 <= ty < GRID_SIZE:
                    if heights[tx, ty] == MAX_BUILDING_HEIGHT:
                        cells.add((int(tx), int(ty)))
        return cells

    def _score_action(
        self,
        heights: np.ndarray,
        move_from: tuple[int, int],
        move_to: tuple[int, int],
        build_on: tuple[int, int],
        threatened: set[tuple[int, int]],
    ) -> float:
        # Winning move dominates everything.
        if heights[move_to] == MAX_BUILDING_HEIGHT:
            return 1000.0

        score = 0.0
        # Block an imminent opponent win by capping the threatened square.
        if build_on in threatened:
            score += 100.0
        elif threatened:
            # A threat exists that this action does not answer.
            score -= 50.0

        # Prefer climbing / staying high.
        score += 5.0 * heights[move_to] - 1.0 * heights[move_from]

        # Avoid building a level-3 square (a build on a level-2 square) unless
        # we could then win there ourselves; the opponent might take it first.
        if heights[build_on] == MAX_BUILDING_HEIGHT - 1 and heights[move_to] < MAX_BUILDING_HEIGHT - 1:
            score -= 20.0

        return score


class ModelOpponent(Opponent):
    """A frozen MaskablePPO snapshot. Samples stochastically so self-play does
    not lock into a single deterministic line."""

    def __init__(self, model_path: Path | str, deterministic: bool = False):
        self.model_path = Path(model_path)
        self.deterministic = deterministic
        self._model = None

    @property
    def model(self):
        if self._model is None:
            # Imported lazily: loading SB3 at module import slows tests/GUI.
            from sb3_contrib import MaskablePPO

            self._model = MaskablePPO.load(self.model_path, device="cpu")
        return self._model

    def choose_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        action, _ = self.model.predict(
            obs, action_masks=mask, deterministic=self.deterministic
        )
        return int(action)


class OpponentPool:
    """Snapshot pool shared between the training callback (writer) and the
    self-play envs (readers). Sampling mixes a random opponent with recent
    frozen snapshots, weighting the newest snapshots highest."""

    def __init__(
        self,
        max_snapshots: int = 5,
        random_prob: float = 0.2,
        decay: float = 0.5,
        rng: np.random.Generator | None = None,
    ):
        self.max_snapshots = max_snapshots
        self.random_prob = random_prob
        self.decay = decay
        self.rng = rng if rng is not None else np.random.default_rng()
        self._snapshots: list[ModelOpponent] = []
        self._random = RandomOpponent(self.rng)

    def add_snapshot(self, model_path: Path | str) -> None:
        self._snapshots.append(ModelOpponent(model_path))
        if len(self._snapshots) > self.max_snapshots:
            self._snapshots.pop(0)

    @property
    def latest(self) -> ModelOpponent | None:
        return self._snapshots[-1] if self._snapshots else None

    def sample(self) -> Opponent:
        if not self._snapshots or self.rng.random() < self.random_prob:
            return self._random
        # Newest snapshot gets weight 1, previous decay, decay^2, ...
        n = len(self._snapshots)
        weights = np.array([self.decay ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()
        idx = int(self.rng.choice(n, p=weights))
        return self._snapshots[idx]
