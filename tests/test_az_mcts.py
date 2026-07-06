"""Tactical tests: MCTS with an untrained net must handle win-in-1 and
loss-in-1 positions through search alone."""

import numpy as np
import torch

from santorini import utils
from santorini.az import fastgame
from santorini.az.config import AZConfig
from santorini.az.mcts import MCTS
from santorini.az.model import AZNet


def make_net() -> AZNet:
    torch.manual_seed(0)
    net = AZNet(channels=16, blocks=1)
    net.eval()
    return net


def make_state(
    workers: list[tuple[int, int]], heights: dict[tuple[int, int], int]
) -> fastgame.State:
    """Playing-phase state; workers in placement order (p0, p1, p0, p1)."""
    state = fastgame.initial_state()
    for pos in workers:
        state = fastgame.step(state, utils.encode_space(pos))
    for (x, y), h in heights.items():
        state.heights[x, y] = h
    return state


def test_win_in_one_is_taken():
    # P0's worker at (3,3) stands on height 2 next to a height-3 at (4,3).
    state = make_state(
        [(3, 3), (0, 4), (0, 0), (4, 0)],
        {(3, 3): 2, (4, 3): 3},
    )
    mcts = MCTS(make_net(), AZConfig(), np.random.default_rng(0))
    ids, visits = mcts.run(state, sims=25)
    action = int(ids[int(np.argmax(visits))])
    _from, move_to, _build = utils.decode_action(action)
    assert move_to == (4, 3)


def test_loss_in_one_is_blocked():
    # P1's worker at (4,4) on height 2 wins next turn via (4,3) at height 3
    # unless P0 caps it. Domes trap the corner workers and funnel P0's live
    # worker at (3,3) so exactly one legal action caps (4,3); cell (2,4)
    # stays open so the block does not stalemate P0 a ply later.
    domes = {
        (0, 1): 4, (1, 1): 4, (2, 0): 4, (2, 1): 4,  # trap corner workers
        (2, 2): 4, (3, 2): 4, (4, 2): 4, (2, 3): 4,
    }
    state = make_state(
        [(3, 3), (4, 4), (0, 0), (1, 0)],
        {**domes, (4, 4): 2, (4, 3): 3},
    )
    actions = fastgame.legal_actions(state)
    blocking = [
        a for a in actions if utils.decode_action(a)[2] == (4, 3)
    ]
    assert len(blocking) == 1
    mcts = MCTS(make_net(), AZConfig(), np.random.default_rng(0))
    ids, visits = mcts.run(state, sims=25)
    action = int(ids[int(np.argmax(visits))])
    _from, _move_to, build_on = utils.decode_action(action)
    assert build_on == (4, 3)


def test_visit_counts_sum_to_sims():
    state = make_state([(1, 1), (3, 3), (2, 2), (4, 0)], {})
    mcts = MCTS(make_net(), AZConfig(), np.random.default_rng(0))
    ids, visits = mcts.run(state, sims=30)
    assert visits.sum() == 30
    assert len(ids) == len(fastgame.legal_actions(state))
    np.testing.assert_array_equal(ids, fastgame.legal_actions(state))


def test_tree_reuse_across_moves():
    state = make_state([(1, 1), (3, 3), (2, 2), (4, 0)], {})
    mcts = MCTS(make_net(), AZConfig(), np.random.default_rng(0))
    ids, visits = mcts.run(state, sims=30)
    action = int(ids[int(np.argmax(visits))])
    nodes_before = len(mcts.nodes)
    state = fastgame.step(state, action)
    mcts.run(state, sims=10)
    assert len(mcts.nodes) > nodes_before  # table kept growing, not rebuilt
    mcts.reset()
    assert not mcts.nodes


def test_mcts_agents_with_sample_moves_vary_games():
    """Net-vs-net matches must not collapse into one repeated game."""
    from santorini.az.agents import MCTSAgent
    from santorini.az.arena import play_game

    net = make_net()

    def game_trace(a, b):
        state = fastgame.initial_state()
        a.reset(), b.reset()
        agents = (a, b)
        trace = []
        while not fastgame.is_terminal(state):
            action = agents[state.player].act(state)
            trace.append(action)
            state = fastgame.step(state, action)
        return tuple(trace)

    rng = np.random.default_rng(0)
    a = MCTSAgent(net, 8, AZConfig(), rng, sample_moves=4)
    b = MCTSAgent(net, 8, AZConfig(), rng, sample_moves=4)
    traces = {game_trace(a, b) for _ in range(4)}
    assert len(traces) > 1

    # Without sampling, play really is deterministic (the failure mode).
    c = MCTSAgent(net, 8, AZConfig(), np.random.default_rng(1))
    d = MCTSAgent(net, 8, AZConfig(), np.random.default_rng(2))
    traces = {game_trace(c, d) for _ in range(3)}
    assert len(traces) == 1
    assert play_game(c, d) in (0, 1)


def test_root_noise_changes_visits():
    state = make_state([(1, 1), (3, 3), (2, 2), (4, 0)], {})
    net = make_net()
    runs = []
    for seed in range(2):
        mcts = MCTS(net, AZConfig(dirichlet_frac=0.9), np.random.default_rng(seed))
        _, visits = mcts.run(state, sims=40, add_noise=True)
        runs.append(visits)
    assert not np.array_equal(runs[0], runs[1])
