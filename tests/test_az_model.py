"""Tests for the policy+value network, in particular the head ordering
(flat output index == action id)."""

import numpy as np
import torch

from santorini import utils
from santorini.az import fastgame
from santorini.az.model import (
    AZNet,
    combined_logits,
    flatten_yx_major,
    load_model,
    masked_log_softmax,
    obs_to_tensor,
    save_model,
)


def playing_state() -> fastgame.State:
    state = fastgame.initial_state()
    for pos in [(1, 1), (3, 3), (2, 2), (4, 0)]:
        state = fastgame.step(state, utils.encode_space(pos))
    return state


def test_flatten_yx_major_mapping():
    """Pin the conv-output -> action-id map: entry (c, x, y) must land at
    flat index (y*5 + x)*C + c, the action encoding of utils.encode_action."""
    c_dim = 64
    x = torch.empty(1, c_dim, 5, 5)
    for c in range(c_dim):
        for bx in range(5):
            for by in range(5):
                x[0, c, bx, by] = (by * 5 + bx) * c_dim + c
    flat = flatten_yx_major(x)[0]
    np.testing.assert_array_equal(flat.numpy(), np.arange(1600))

    # Placement head: C=1, flat index = space index y*5 + x.
    p = torch.empty(1, 1, 5, 5)
    for bx in range(5):
        for by in range(5):
            p[0, 0, bx, by] = utils.encode_space((bx, by))
    np.testing.assert_array_equal(flatten_yx_major(p)[0].numpy(), np.arange(25))


def test_forward_shapes_and_value_range():
    net = AZNet(channels=16, blocks=1)
    obs = np.stack([fastgame.canonical_obs(playing_state())] * 3)
    move, place, value = net(obs_to_tensor(obs))
    assert move.shape == (3, 1600)
    assert place.shape == (3, 25)
    assert value.shape == (3,)
    assert torch.all(value.abs() <= 1)


def test_save_load_roundtrip(tmp_path):
    net = AZNet(channels=16, blocks=1)
    net.eval()
    path = tmp_path / "net.pt"
    save_model(net, path, extra={"iteration": 7})
    loaded = load_model(path)
    obs = fastgame.canonical_obs(playing_state())
    logits_a, v_a = net.predict(obs, setup=False)
    logits_b, v_b = loaded.predict(obs, setup=False)
    np.testing.assert_allclose(logits_a, logits_b, atol=1e-6)
    assert v_a == v_b


def _overfit_one(state: fastgame.State, target_action: int) -> int:
    """Train a tiny net on a single (obs, one-hot pi, z) example; return the
    argmax of the masked logits afterwards."""
    torch.manual_seed(0)
    net = AZNet(channels=16, blocks=1)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
    obs = obs_to_tensor(fastgame.canonical_obs(state)).unsqueeze(0)
    mask = torch.from_numpy(fastgame.legal_mask(state)).unsqueeze(0)
    setup = torch.tensor([fastgame.is_setup(state)])
    pi = torch.zeros(1, 1600)
    pi[0, target_action] = 1.0
    z = torch.tensor([1.0])
    for _ in range(60):
        opt.zero_grad()
        move, place, value = net(obs)
        logits = combined_logits(move, place, setup)
        policy_loss = -(pi * masked_log_softmax(logits, mask)).sum(dim=-1).mean()
        loss = policy_loss + torch.nn.functional.mse_loss(value, z)
        assert torch.isfinite(loss), "loss went non-finite during overfit"
        loss.backward()
        opt.step()
    net.eval()
    with torch.inference_mode():
        move, place, _ = net(obs)
        logits = combined_logits(move, place, setup)
        logits = logits.masked_fill(~mask, float("-inf"))
    return int(logits[0].argmax())


def test_head_ordering_overfit_playing_phase():
    state = playing_state()
    target = fastgame.legal_actions(state)[7]
    assert _overfit_one(state, target) == target


def test_head_ordering_overfit_setup_phase():
    state = fastgame.step(fastgame.initial_state(), utils.encode_space((2, 2)))
    target = utils.encode_space((1, 3))
    assert target in fastgame.legal_actions(state)
    assert _overfit_one(state, target) == target
