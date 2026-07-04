"""Tests for opponent policies."""

import numpy as np

from santorini import utils
from santorini.game import Game, GameState
from santorini.opponents import GreedyOpponent, OpponentPool, RandomOpponent
from santorini.selfplay_env import NUM_ACTIONS


def make_game(placements: list[tuple[int, int]], heights: dict[tuple[int, int], int]) -> Game:
    """Build a 2-player game in PLAYING state. `placements` lists worker
    positions in placement order (p0, p1, p0, p1); `heights` sets building
    heights afterwards."""
    game = Game()
    game.step(2)
    for position in placements:
        game.step(utils.encode_space(position))
    assert game.state == GameState.PLAYING
    for position, height in heights.items():
        for _ in range(height):
            game.board.build(position)
    game._update_valid_actions()
    return game


def obs_and_mask(game: Game, player_idx: int) -> tuple[np.ndarray, np.ndarray]:
    obs = game.board.get_observation(player_idx)
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
    mask[list(game.valid_actions)] = 1
    return obs, mask


def test_random_opponent_picks_valid_action():
    game = make_game([(0, 0), (4, 4), (0, 4), (4, 0)], {})
    obs, mask = obs_and_mask(game, 0)
    opp = RandomOpponent(np.random.default_rng(0))
    for _ in range(20):
        assert mask[opp.choose_action(obs, mask)] == 1


def test_greedy_takes_winning_move():
    # Player 0's worker at (0, 0) stands on height 2; (0, 1) is height 3.
    game = make_game(
        [(0, 0), (4, 4), (2, 2), (4, 0)],
        {(0, 0): 2, (0, 1): 3},
    )
    game._update_valid_actions()
    obs, mask = obs_and_mask(game, 0)
    opp = GreedyOpponent(np.random.default_rng(0))
    action = opp.choose_action(obs, mask)
    _move_from, move_to, _build_on = utils.decode_action(action)
    assert move_to == (0, 1)


def test_greedy_blocks_opponent_win():
    # It is player 0's turn. Player 1's worker at (4, 4) stands on height 2
    # next to a height-3 tower at (4, 3): player 1 wins next turn unless the
    # tower is capped. Player 0's workers are adjacent enough to build there.
    game = make_game(
        [(3, 3), (4, 4), (0, 0), (4, 0)],
        {(4, 4): 2, (4, 3): 3},
    )
    game._update_valid_actions()
    obs, mask = obs_and_mask(game, 0)
    opp = GreedyOpponent(np.random.default_rng(0))
    action = opp.choose_action(obs, mask)
    _move_from, _move_to, build_on = utils.decode_action(action)
    assert build_on == (4, 3)


def test_greedy_places_centrally_during_setup():
    game = Game()
    game.step(2)
    obs = game.board.get_observation(0)
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
    mask[list(game.valid_actions)] = 1
    opp = GreedyOpponent(np.random.default_rng(0))
    action = opp.choose_action(obs, mask)
    assert utils.decode_space(action) == (2, 2)


def test_opponent_pool_random_when_empty():
    pool = OpponentPool(rng=np.random.default_rng(0))
    assert isinstance(pool.sample(), RandomOpponent)
    assert pool.latest is None


def test_opponent_pool_caps_snapshots():
    pool = OpponentPool(max_snapshots=2, rng=np.random.default_rng(0))
    for i in range(4):
        pool.add_snapshot(f"fake_{i}.zip")
    assert len(pool._snapshots) == 2
    assert pool.latest.model_path.name == "fake_3.zip"


def test_opponent_pool_samples_snapshots():
    pool = OpponentPool(random_prob=0.0, rng=np.random.default_rng(0))
    pool.add_snapshot("fake_0.zip")
    sampled = pool.sample()
    assert sampled is pool.latest
