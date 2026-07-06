"""Self-play example generation and replay buffer tests."""

import numpy as np
import torch

from santorini.az import fastgame
from santorini.az.config import AZConfig
from santorini.az.model import AZNet
from santorini.az.replay import ReplayBuffer
from santorini.az.selfplay import play_game


def tiny_config() -> AZConfig:
    return AZConfig(sims=8)


def make_net() -> AZNet:
    torch.manual_seed(0)
    net = AZNet(channels=16, blocks=1)
    net.eval()
    return net


def test_play_game_examples():
    examples, stats = play_game(make_net(), tiny_config(), np.random.default_rng(0))
    assert stats.winner in (0, 1)
    assert len(examples) == stats.plies
    assert stats.mean_root_entropy > 0

    # Placement plies come first, then playing plies.
    assert [ex.placed for ex in examples[:4]] == [0, 1, 2, 3]
    assert all(ex.placed == 4 for ex in examples[4:])

    for ex in examples:
        assert ex.pi_ids.dtype == np.int16
        np.testing.assert_allclose(ex.pi_probs.sum(), 1.0, rtol=1e-5)
        state = fastgame.State(ex.heights, ex.workers, ex.placed, ex.player, -1)
        np.testing.assert_array_equal(np.sort(ex.pi_ids), fastgame.legal_actions(state))

    # Players strictly alternate every ply (including placement), so z must
    # strictly alternate sign, ending at +1 for the winner's last move.
    zs = [ex.z for ex in examples]
    assert set(zs) == {1.0, -1.0}
    assert all(a == -b for a, b in zip(zs, zs[1:]))
    assert zs[-1] == 1.0
    assert examples[-1].player == stats.winner


def test_replay_buffer_sample_shapes():
    net = make_net()
    rng = np.random.default_rng(1)
    buf = ReplayBuffer(capacity=10_000)
    for seed in range(2):
        examples, _ = play_game(net, tiny_config(), np.random.default_rng(seed))
        buf.add(examples)
    batch = buf.sample(32, rng)
    assert batch["obs"].shape == (32, 5, 5, 13)
    assert batch["pi"].shape == (32, 1600)
    assert batch["mask"].shape == (32, 1600)
    np.testing.assert_allclose(batch["pi"].sum(axis=1), 1.0, rtol=1e-5)
    # pi mass only on legal (masked) entries; setup rows only in 0..24.
    assert not (batch["pi"] * ~batch["mask"]).any()
    assert not batch["pi"][batch["setup"], 25:].any()
    assert set(np.unique(batch["z"])) <= {-1.0, 1.0}


def test_replay_buffer_ring_capacity():
    buf = ReplayBuffer(capacity=50)
    examples, _ = play_game(make_net(), tiny_config(), np.random.default_rng(0))
    while len(buf) < 50:
        buf.add(examples)
    buf.add(examples[:10])
    assert len(buf) == 50


def test_replay_buffer_save_load_roundtrip(tmp_path):
    buf = ReplayBuffer(capacity=1000)
    examples, _ = play_game(make_net(), tiny_config(), np.random.default_rng(0))
    buf.add(examples)
    path = tmp_path / "buffer.npz"
    buf.save(path)
    loaded = ReplayBuffer.load(path)
    assert len(loaded) == len(buf)
    assert loaded.capacity == buf.capacity
    for a, b in zip(buf._data, loaded._data):
        np.testing.assert_array_equal(a.heights, b.heights)
        np.testing.assert_array_equal(a.workers, b.workers)
        np.testing.assert_array_equal(a.pi_ids, b.pi_ids)
        np.testing.assert_allclose(a.pi_probs, b.pi_probs)
        assert (a.placed, a.player, a.z) == (b.placed, b.player, b.z)

    # Sampling from the reloaded buffer works identically.
    rng = np.random.default_rng(2)
    batch = loaded.sample(8, rng)
    assert batch["obs"].shape == (8, 5, 5, 13)
