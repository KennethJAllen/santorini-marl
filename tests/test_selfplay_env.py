"""Tests for the single-agent self-play environment."""

import numpy as np
import pytest

from santorini.game import GameState
from santorini.opponents import GreedyOpponent, RandomOpponent
from santorini.selfplay_env import NUM_ACTIONS, SantoriniSelfPlayEnv


def run_episode(env: SantoriniSelfPlayEnv, seed: int, rng: np.random.Generator):
    """Play one episode with uniformly random learner actions.
    Returns (total_reward, final_info)."""
    env.reset(seed=seed)
    total_reward = 0.0
    terminated = False
    info = {}
    for _ in range(500):
        mask = env.action_masks()
        action = int(rng.choice(np.flatnonzero(mask)))
        _obs, reward, terminated, _truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            break
    assert terminated, "episode did not terminate"
    return total_reward, info


def test_reset_returns_learner_turn_obs():
    for seat in (0, 1):
        env = SantoriniSelfPlayEnv(
            opponent=RandomOpponent(np.random.default_rng(0)), learner_seat=seat
        )
        obs, info = env.reset(seed=0)
        assert obs.shape == (5, 5, 11)
        assert info["learner_seat"] == seat
        assert env.game.current_player_idx == seat
        # Seat 1 moves second: the opponent has already placed one worker.
        placed = obs[:, :, 9].sum() + obs[:, :, 10].sum()
        assert placed == (1 if seat == 1 else 0)


def test_action_mask_matches_game_valid_actions():
    env = SantoriniSelfPlayEnv(opponent=RandomOpponent(np.random.default_rng(0)))
    env.reset(seed=3)
    mask = env.action_masks()
    assert mask.shape == (NUM_ACTIONS,)
    assert set(np.flatnonzero(mask)) == env.game.valid_actions


def test_terminal_rewards_are_win_loss():
    """Both +1 (learner wins) and -1 (learner loses) terminals must be
    delivered — the old shared-stream wrapper silently dropped the -1."""
    env = SantoriniSelfPlayEnv(opponent=RandomOpponent(np.random.default_rng(1)))
    rng = np.random.default_rng(2)
    outcomes = set()
    for seed in range(30):
        total_reward, info = run_episode(env, seed, rng)
        assert total_reward in (1.0, -1.0)
        assert info["terminal_reward"] == total_reward
        assert info["learner_won"] == (total_reward == 1.0)
        assert env.game.state == GameState.GAME_OVER
        outcomes.add(total_reward)
    assert outcomes == {1.0, -1.0}


def test_learner_loses_to_stronger_opponent_sometimes():
    """A random learner vs the greedy opponent must receive -1 terminals."""
    env = SantoriniSelfPlayEnv(opponent=GreedyOpponent(np.random.default_rng(0)))
    rng = np.random.default_rng(0)
    rewards = [run_episode(env, seed, rng)[0] for seed in range(10)]
    assert -1.0 in rewards


def test_seat_randomized_by_default():
    env = SantoriniSelfPlayEnv(opponent=RandomOpponent(np.random.default_rng(0)))
    seats = set()
    for seed in range(20):
        _obs, info = env.reset(seed=seed)
        seats.add(info["learner_seat"])
    assert seats == {0, 1}


def test_shaping_preserves_terminal_reward():
    env = SantoriniSelfPlayEnv(
        opponent=RandomOpponent(np.random.default_rng(0)),
        shaping_scale=0.5,
        gamma=0.99,
    )
    rng = np.random.default_rng(4)
    for seed in range(5):
        total_reward, info = run_episode(env, seed, rng)
        assert info["terminal_reward"] in (1.0, -1.0)
        # Total = terminal + telescoped potential shaping; with Phi(terminal)=0
        # and Phi(start)=0 the shaping sum is bounded and small.
        assert abs(total_reward - info["terminal_reward"]) == pytest.approx(
            abs(info["shaping_total"]), abs=1e-9
        )


def test_placement_actions_valid_during_setup():
    env = SantoriniSelfPlayEnv(
        opponent=RandomOpponent(np.random.default_rng(0)), learner_seat=0
    )
    env.reset(seed=0)
    assert env.game.state == GameState.SETUP
    mask = env.action_masks()
    # All placement actions are board-space indices < 25.
    assert np.flatnonzero(mask).max() < 25
