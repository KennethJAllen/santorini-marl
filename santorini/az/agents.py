"""Agents over fastgame states: MCTS, raw policy, and adapters wrapping the
existing santorini.opponents heuristics.

All agents expose `act(state) -> int` and `reset()` (called between games).
"""

from __future__ import annotations

import numpy as np

from santorini.az import fastgame
from santorini.az.config import AZConfig
from santorini.az.mcts import MCTS
from santorini.az.model import AZNet
from santorini.opponents import GreedyOpponent, RandomOpponent


class MCTSAgent:
    """Plays the argmax-visits move of a PUCT search (no root noise).

    With `sample_moves > 0` the agent samples its first N own moves from the
    visit distribution instead of taking the argmax. Without this, two MCTS
    agents play a fully deterministic game, so a many-game match degenerates
    into the same two games (one per seat) replayed — arena winrates become
    0/50/100%. Use it for any net-vs-net match.
    """

    def __init__(
        self,
        net: AZNet,
        sims: int,
        config: AZConfig | None = None,
        rng: np.random.Generator | None = None,
        sample_moves: int = 0,
    ):
        self.sims = sims
        self.mcts = MCTS(net, config, rng)
        self.sample_moves = sample_moves
        self._own_ply = 0

    def reset(self) -> None:
        self.mcts.reset()
        self._own_ply = 0

    def act(self, state: fastgame.State) -> int:
        ids, visits = self.mcts.run(state, self.sims, add_noise=False)
        if self._own_ply < self.sample_moves:
            pi = visits.astype(np.float64) / visits.sum()
            action = int(self.mcts.rng.choice(ids, p=pi))
        else:
            action = int(ids[int(np.argmax(visits))])
        self._own_ply += 1
        return action


class RawPolicyAgent:
    """Argmax of the masked policy logits, no search: measures what the net
    itself has learned as opposed to what search adds."""

    def __init__(self, net: AZNet):
        self.net = net

    def reset(self) -> None:
        pass

    def act(self, state: fastgame.State) -> int:
        ids = np.asarray(fastgame.legal_actions(state))
        logits, _ = self.net.predict(
            fastgame.canonical_obs(state), fastgame.is_setup(state)
        )
        return int(ids[int(np.argmax(logits[ids]))])


class _OpponentAdapter:
    """Feeds an existing Opponent policy the (5,5,11) perspective obs and
    flat mask it expects, sourced from a fastgame state."""

    def __init__(self, opponent):
        self.opponent = opponent

    def reset(self) -> None:
        pass

    def act(self, state: fastgame.State) -> int:
        obs = fastgame.canonical_obs(state)[:, :, :11]
        mask = fastgame.legal_mask(state)
        return int(self.opponent.choose_action(obs, mask))


class RandomAgent(_OpponentAdapter):
    def __init__(self, rng: np.random.Generator | None = None):
        super().__init__(RandomOpponent(rng))


class GreedyAgent(_OpponentAdapter):
    def __init__(self, rng: np.random.Generator | None = None):
        super().__init__(GreedyOpponent(rng))
