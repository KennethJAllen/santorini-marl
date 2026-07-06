"""Single-agent Gymnasium environment for frozen-opponent self-play.

The learner controls one seat (randomized per episode by default); the
opponent's turns are played inside `reset()`/`step()` by a frozen policy
sampled from an `OpponentPool`. This makes the learner's MDP well-defined:
rewards are +1 win / -1 loss delivered to the learner, and value
bootstrapping always happens on the learner's own next decision point —
unlike the shared-stream AEC wrapper, which silently dropped the loser's
terminal reward and bootstrapped values across player perspectives.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from santorini.config import GRID_SIZE, NUM_WORKERS
from santorini.game import Game, GameState
from santorini.opponents import Opponent, OpponentPool, RandomOpponent

NUM_ACTIONS = GRID_SIZE * GRID_SIZE * 8 * 8
NUM_PLAYERS = 2


class SantoriniSelfPlayEnv(gym.Env):
    """Santorini from the perspective of a single learning agent."""

    metadata = {"name": "santorini_selfplay_v1", "render_modes": ["ansi"]}

    def __init__(
        self,
        opponent_pool: OpponentPool | None = None,
        opponent: Opponent | None = None,
        learner_seat: int | None = None,
        shaping_scale: float = 0.0,
        gamma: float = 0.99,
    ):
        """
        opponent_pool: pool to sample an opponent from at each reset.
        opponent: fixed opponent (overrides the pool); used for evaluation.
        learner_seat: fix the learner to seat 0 or 1; random each episode if None.
        shaping_scale: scale of the optional potential-based shaping term
            (0 disables it; the terminal reward stays +1/-1 either way).
        gamma: discount used in the potential-based shaping term; should match
            the agent's discount so shaping does not change the optimal policy.
        """
        super().__init__()
        self.game = Game()
        self.opponent_pool = opponent_pool
        self.fixed_opponent = opponent
        self.fixed_seat = learner_seat
        self.shaping_scale = shaping_scale
        self.gamma = gamma

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(GRID_SIZE, GRID_SIZE, 11), dtype=np.int8
        )

        self.learner_seat = 0
        self._opponent: Opponent = opponent or RandomOpponent()
        self._ep_len = 0
        self._shaping_total = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.learner_seat = (
            self.fixed_seat
            if self.fixed_seat is not None
            else int(self.np_random.integers(NUM_PLAYERS))
        )
        if self.fixed_opponent is not None:
            self._opponent = self.fixed_opponent
        elif self.opponent_pool is not None:
            self._opponent = self.opponent_pool.sample()
        else:
            self._opponent = RandomOpponent(self.np_random)

        self._ep_len = 0
        self._shaping_total = 0.0

        self.game.reset()
        self.game.step(NUM_PLAYERS)  # PLAYER_SELECT -> SETUP
        self._play_opponent_turns()
        return self._observe(), {"learner_seat": self.learner_seat}

    def step(self, action):
        phi_before = self._potential()

        self.game.step(int(action))
        self._ep_len += 1
        if not self.game.is_done():
            self._play_opponent_turns()

        terminated = self.game.is_done()

        shaping = 0.0
        if self.shaping_scale:
            # Potential-based shaping with Phi(terminal) = 0.
            phi_after = 0.0 if terminated else self._potential()
            shaping = self.shaping_scale * (self.gamma * phi_after - phi_before)
            self._shaping_total += shaping

        reward = shaping
        info = {}
        if terminated:
            learner_won = self.game.winner is self.game.players[self.learner_seat]
            terminal_reward = 1.0 if learner_won else -1.0
            reward += terminal_reward
            info = {
                "learner_won": learner_won,
                "learner_seat": self.learner_seat,
                "learner_ep_len": self._ep_len,
                "terminal_reward": terminal_reward,
                "shaping_total": self._shaping_total,
            }

        return self._observe(), reward, terminated, False, info

    def action_masks(self) -> np.ndarray:
        """Valid-action mask for the learner (MaskablePPO hook)."""
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        mask[list(self.game.valid_actions)] = True
        return mask

    def render(self):
        return str(self.game.board)

    def _observe(self) -> np.ndarray:
        return self.game.board.get_observation(self.learner_seat)

    def _play_opponent_turns(self) -> None:
        """Advance the game until it is the learner's turn or the game ends."""
        opp_seat = 1 - self.learner_seat
        while (
            not self.game.is_done()
            and self.game.current_player_idx != self.learner_seat
        ):
            obs = self.game.board.get_observation(opp_seat)
            mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
            mask[list(self.game.valid_actions)] = 1
            action = self._opponent.choose_action(obs, mask)
            self.game.step(action)

    def _potential(self) -> float:
        """Phi(s) = max learner worker height - max opponent worker height."""
        if self.game.state != GameState.PLAYING:
            return 0.0
        heights = []
        for player in self.game.players:
            positions = [w.position for w in player.workers if w.position is not None]
            if len(positions) < NUM_WORKERS:
                return 0.0
            heights.append(max(self.game.board.get_height(p) for p in positions))
        return float(heights[self.learner_seat] - heights[1 - self.learner_seat])
