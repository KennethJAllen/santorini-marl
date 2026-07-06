"""Cross-validation of the fast numpy game core against the reference Game.

The fuzz tests run both engines in lockstep and assert identical legal-action
sets, heights, worker positions, and winner at every ply. If these pass, the
fast core is trusted as a drop-in simulator for MCTS.
"""

import numpy as np
import pytest

from santorini import utils
from santorini.az import fastgame
from santorini.game import Game, GameState


def game_workers(game: Game) -> np.ndarray:
    workers = np.full((2, 2, 2), -1, dtype=np.int8)
    for p, pl in enumerate(game.players):
        for w in pl.workers:
            workers[p, w.get_id()] = w.position
    return workers


def game_heights(game: Game) -> np.ndarray:
    heights = np.zeros((5, 5), dtype=np.int8)
    for x in range(5):
        for y in range(5):
            heights[x, y] = game.board.get_height((x, y))
    return heights


def assert_synced(game: Game, state: fastgame.State, ply: int) -> None:
    assert sorted(game.valid_actions) == fastgame.legal_actions(state), f"ply {ply}"
    np.testing.assert_array_equal(game_heights(game), state.heights, err_msg=f"ply {ply}")
    np.testing.assert_array_equal(game_workers(game), state.workers, err_msg=f"ply {ply}")
    game_winner = game.winner.get_id() if game.winner is not None else -1
    assert game_winner == state.winner, f"ply {ply}"
    if not game.is_done():
        assert game.current_player_idx == state.player, f"ply {ply}"


def run_lockstep(game: Game, state: fastgame.State, rng: np.random.Generator) -> None:
    """Play random moves on both engines until the game ends."""
    ply = 0
    assert_synced(game, state, ply)
    while not game.is_done() and game.valid_actions:
        action = int(rng.choice(sorted(game.valid_actions)))
        game.step(action)
        state = fastgame.step(state, action)
        ply += 1
        assert_synced(game, state, ply)
        # ch0-10 of the canonical obs must match the reference observation.
        if not game.is_done() and ply % 7 == 0:
            obs = fastgame.canonical_obs(state)
            ref = game.board.get_observation(state.player)
            np.testing.assert_array_equal(obs[:, :, :11].astype(np.int8), ref)
    assert game.is_done() == fastgame.is_terminal(state)


def test_lockstep_fuzz_from_initial():
    rng = np.random.default_rng(0)
    for _ in range(200):
        game = Game()
        game.step(2)
        run_lockstep(game, fastgame.initial_state(), rng)


def test_lockstep_fuzz_from_prebuilt_heights():
    """Seed games with random placements plus random pre-built heights (the
    make_game pattern from test_opponents), then play out in lockstep."""
    rng = np.random.default_rng(1)
    for _ in range(100):
        game = Game()
        game.step(2)
        cells = [(x, y) for x in range(5) for y in range(5)]
        picks = rng.choice(25, size=4, replace=False)
        for idx in picks:
            game.step(utils.encode_space(cells[idx]))
        assert game.state == GameState.PLAYING
        n_builds = int(rng.integers(0, 30))
        for _ in range(n_builds):
            pos = cells[int(rng.integers(25))]
            if game.board.get_height(pos) < 4:
                game.board.build(pos)
        game._update_valid_actions()
        run_lockstep(game, fastgame.from_game(game), rng)


def test_winning_move_allows_any_adjacent_build():
    """A move onto height 3 wins; the build part may target occupied or
    capped cells and no build is applied."""
    state = fastgame.initial_state()
    for pos in [(0, 0), (4, 4), (1, 1), (4, 0)]:
        state = fastgame.step(state, utils.encode_space(pos))
    state.heights[0, 0] = 2
    state.heights[0, 1] = 3
    state.heights[1, 0] = 4  # capped cell adjacent to the winning square

    # Move from (0,0) south to (0,1): wins. Build part N of (0,1) is the
    # capped (1,0)? No: direction N from (0,1) is (0,0) — the vacated square;
    # NE is (1,0) the dome. All on-board build dirs must be legal.
    legal = set(fastgame.legal_actions(state))
    move_dir = utils.DIRS.index((0, 1))  # S
    base = (0 * 5 + 0) * 64 + move_dir * 8
    for bdir, (bdx, bdy) in enumerate(utils.DIRS):
        bx, by = 0 + bdx, 1 + bdy
        expected = 0 <= bx < 5 and 0 <= by < 5
        assert ((base + bdir) in legal) == expected

    win_action = base + utils.DIRS.index((1, -1))  # "build" on the dome at (1,0)
    heights_before = state.heights.copy()
    after = fastgame.step(state, win_action)
    assert after.winner == 0
    np.testing.assert_array_equal(after.heights, heights_before)  # no build applied
    assert fastgame.terminal_value(after) == -1.0  # player to move lost


def test_build_on_vacated_square_is_legal():
    state = fastgame.initial_state()
    for pos in [(2, 2), (4, 4), (0, 0), (4, 0)]:
        state = fastgame.step(state, utils.encode_space(pos))
    # Move worker from (2,2) N to (2,1), build S back onto (2,2).
    action = utils.encode_action(((2, 2), (2, 1), (2, 2)))
    assert action in fastgame.legal_actions(state)
    after = fastgame.step(state, action)
    assert after.heights[2, 2] == 1
    assert after.player == 1


def test_own_other_worker_blocks_build():
    state = fastgame.initial_state()
    for pos in [(2, 2), (4, 4), (2, 0), (4, 0)]:
        state = fastgame.step(state, utils.encode_space(pos))
    # Worker at (2,2) moves N to (2,1); building further N onto (2,0) hits
    # player 0's other worker: illegal.
    action = utils.encode_action(((2, 2), (2, 1), (2, 0)))
    assert action not in fastgame.legal_actions(state)


def test_stalemate_previous_player_wins():
    """Trap player 1's workers so they have no moves: player 0 wins."""
    state = fastgame.initial_state()
    for pos in [(2, 2), (0, 0), (2, 3), (1, 0)]:
        state = fastgame.step(state, utils.encode_space(pos))
    # Wall in the corner cluster around p1's workers at (0,0) and (1,0).
    for cell in [(0, 1), (1, 1), (2, 0), (2, 1)]:
        state.heights[cell] = 4

    # Mirror the position in the reference engine.
    game = Game()
    game.step(2)
    for pos in [(2, 2), (0, 0), (2, 3), (1, 0)]:
        game.step(utils.encode_space(pos))
    for cell in [(0, 1), (1, 1), (2, 0), (2, 1)]:
        for _ in range(4):
            game.board.build(cell)
    game._update_valid_actions()

    assert sorted(game.valid_actions) == fastgame.legal_actions(state)
    # Any p0 action that doesn't open an escape ends the game with p0 winning.
    action = utils.encode_action(((2, 2), (3, 2), (4, 2)))
    game.step(action)
    after = fastgame.step(state, action)
    assert game.is_done() and game.winner.get_id() == 0
    assert after.winner == 0
    assert fastgame.terminal_value(after) == -1.0


def test_threat_planes():
    state = fastgame.initial_state()
    for pos in [(0, 0), (4, 4), (2, 2), (4, 0)]:
        state = fastgame.step(state, utils.encode_space(pos))
    state.heights[0, 0] = 2  # my worker standing at level 2
    state.heights[0, 1] = 3  # empty level-3 next to it: my threat
    state.heights[1, 1] = 3  # empty level-3 next to it: my threat
    state.heights[4, 3] = 3  # next to opp worker at (4,4) but opp is at level 0
    state.heights[3, 1] = 3  # level-3 not adjacent to any worker

    obs = fastgame.canonical_obs(state)
    mine = {(0, 1), (1, 1)}
    for x in range(5):
        for y in range(5):
            assert obs[x, y, 11] == ((x, y) in mine)
            assert obs[x, y, 12] == 0  # opponent has no threats

    # Occupied level-3 cells are not threats (exact-legality variant).
    state2 = state.copy()
    state2.workers[1, 1] = (1, 1)  # opp worker occupies one threat square
    obs2 = fastgame.canonical_obs(state2)
    assert obs2[1, 1, 11] == 0
    assert obs2[0, 1, 11] == 1

    # From the opponent's perspective the same threats appear on ch12.
    state3 = state.copy()
    state3.player = 1
    obs3 = fastgame.canonical_obs(state3)
    for x, y in mine:
        assert obs3[x, y, 12] == 1
    assert obs3[:, :, 11].sum() == 0


def test_key_distinguishes_player_to_move():
    state = fastgame.initial_state()
    for pos in [(0, 0), (4, 4), (2, 2), (4, 0)]:
        state = fastgame.step(state, utils.encode_space(pos))
    other = state.copy()
    other.player = 1
    assert fastgame.key(state) != fastgame.key(other)


def test_step_rejects_terminal_state():
    state = fastgame.initial_state()
    state.winner = 0
    with pytest.raises(ValueError):
        fastgame.step(state, 0)
