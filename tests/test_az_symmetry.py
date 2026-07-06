"""Property tests for the D4 symmetry tables."""

import numpy as np

from santorini import utils
from santorini.az import fastgame, symmetry
from santorini.az.symmetry import (
    ACTION_PERM,
    DIR_PERM,
    NUM_TRANSFORMS,
    SPACE_PERM,
    apply_xy,
)


def test_perms_are_bijections():
    for t in range(NUM_TRANSFORMS):
        for perm, n in ((SPACE_PERM[t], 25), (DIR_PERM[t], 8), (ACTION_PERM[t], 1600)):
            assert sorted(perm.tolist()) == list(range(n))


def test_identity_transform():
    np.testing.assert_array_equal(SPACE_PERM[0], np.arange(25))
    np.testing.assert_array_equal(ACTION_PERM[0], np.arange(1600))


def test_group_closure():
    """Composing any two of the 8 space perms yields one of the 8."""
    perms = {tuple(SPACE_PERM[t].tolist()) for t in range(NUM_TRANSFORMS)}
    assert len(perms) == NUM_TRANSFORMS
    for a in range(NUM_TRANSFORMS):
        for b in range(NUM_TRANSFORMS):
            composed = tuple(SPACE_PERM[a][SPACE_PERM[b]].tolist())
            assert composed in perms


def test_action_perm_matches_decode_oracle():
    """decode(ACTION_PERM[t][a]) must equal the coordinate-mapped decode(a).

    decode_action can return off-board move/build coords; apply_xy is affine
    so the oracle holds for those too.
    """
    rng = np.random.default_rng(0)
    for a in rng.integers(0, 1600, size=200):
        expected_parts = utils.decode_action(int(a))
        for t in range(NUM_TRANSFORMS):
            got = utils.decode_action(int(ACTION_PERM[t][a]))
            expected = tuple(apply_xy(t, x, y) for x, y in expected_parts)
            assert got == expected, (t, int(a))


def _random_reachable_states(rng: np.random.Generator, n_games: int):
    """Yield states along random playouts, both setup and playing phases."""
    for _ in range(n_games):
        state = fastgame.initial_state()
        while not fastgame.is_terminal(state):
            yield state
            actions = fastgame.legal_actions(state)
            state = fastgame.step(state, int(rng.choice(actions)))


def test_equivariance_of_legal_mask_and_obs():
    rng = np.random.default_rng(2)
    checked_setup = checked_playing = 0
    for state in _random_reachable_states(rng, 8):
        mask = fastgame.legal_mask(state)
        obs = fastgame.canonical_obs(state)
        setup = fastgame.is_setup(state)
        for t in range(NUM_TRANSFORMS):
            tstate = fastgame.transform_state(state, t)
            tmask = fastgame.legal_mask(tstate)
            if setup:
                # Placement actions are space indices 0..24.
                np.testing.assert_array_equal(tmask[SPACE_PERM[t]][:25], mask[:25])
                assert not tmask[25:].any()
            else:
                np.testing.assert_array_equal(tmask[ACTION_PERM[t]], mask)
            np.testing.assert_array_equal(
                symmetry.transform_obs(obs, t), fastgame.canonical_obs(tstate)
            )
        if setup:
            checked_setup += 1
        else:
            checked_playing += 1
    assert checked_setup >= 8 and checked_playing >= 8


def test_transform_pi_sparse_dense_agree():
    rng = np.random.default_rng(3)
    state = fastgame.initial_state()
    for pos in [(0, 0), (4, 4), (2, 2), (4, 0)]:
        state = fastgame.step(state, utils.encode_space(pos))
    ids = np.array(fastgame.legal_actions(state))
    probs = rng.dirichlet(np.ones(len(ids)))
    dense = np.zeros(1600)
    dense[ids] = probs
    for t in range(NUM_TRANSFORMS):
        tdense = symmetry.transform_pi(dense, t, setup=False)
        tids = symmetry.transform_action_ids(ids, t, setup=False)
        np.testing.assert_allclose(tdense[tids], probs)
        np.testing.assert_allclose(tdense.sum(), dense.sum())
