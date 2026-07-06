"""Residual CNN policy+value network.

Input is the (5,5,13) canonical observation transposed to NCHW (13,5,5), so
the tensor's H axis is board x and W axis is board y. Both policy heads
flatten y-major so that flat output index == action id:

- move head: 64 channels, one per (move_dir*8 + build_dir); flat index
  (y*5 + x)*64 + move*8 + build.
- placement head: 1 channel; flat index y*5 + x = space index. Its 25 logits
  replace move-head logits 0..24 during SETUP (placement actions share those
  indices in the Discrete(1600) space).

Masking happens on raw logits (illegal entries filled with -inf before
softmax), so no distribution validation issues arise downstream.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from santorini.az.fastgame import NUM_SPACES, OBS_CHANNELS


def flatten_yx_major(x: torch.Tensor) -> torch.Tensor:
    """(B, C, X, Y) -> (B, Y*X*C) with index (y*5 + x)*C + c.

    This is the load-bearing map from conv output to action ids; pinned
    directly by tests to catch x/y transposition.
    """
    return x.permute(0, 3, 2, 1).reshape(x.shape[0], -1)


def obs_to_tensor(obs: np.ndarray) -> torch.Tensor:
    """(..., 5, 5, 13) float32 obs -> (..., 13, 5, 5) tensor."""
    arr = np.ascontiguousarray(np.moveaxis(obs, -1, -3))
    return torch.from_numpy(arr)


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Log-softmax over legal entries; illegal entries are zeroed (not -inf)
    so that cross-entropy against a target with zero mass there stays finite
    (0 * -inf would be nan)."""
    logp = F.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
    return logp.masked_fill(~mask, 0.0)


def combined_logits(
    move: torch.Tensor, place: torch.Tensor, setup: torch.Tensor
) -> torch.Tensor:
    """Merge the two policy heads into one (B, 1600) logit tensor: rows in
    SETUP get their first 25 entries from the placement head."""
    logits = move.clone()
    if setup.any():
        logits[setup, :NUM_SPACES] = place[setup]
    return logits


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class AZNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 4):
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.stem = nn.Sequential(
            nn.Conv2d(OBS_CHANNELS, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.torso = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.move_head = nn.Conv2d(channels, 64, 1)
        self.place_head = nn.Conv2d(channels, 1, 1)
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(32 * NUM_SPACES, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.torso(self.stem(x))
        move = flatten_yx_major(self.move_head(h))
        place = flatten_yx_major(self.place_head(h))
        value = self.value_fc(self.value_conv(h).flatten(1)).squeeze(-1)
        return move, place, value

    @torch.inference_mode()
    def predict(self, obs: np.ndarray, setup: bool) -> tuple[np.ndarray, float]:
        """Single-state inference for MCTS: (1600,) logits + value.

        During SETUP the first 25 logits come from the placement head; the
        caller only ever reads legal indices, which are then <= 24.
        """
        if self.training:
            self.eval()
        move, place, value = self(obs_to_tensor(obs).unsqueeze(0))
        logits = move[0].numpy()
        if setup:
            logits[:NUM_SPACES] = place[0].numpy()
        return logits, float(value[0])


def save_model(net: AZNet, path: Path | str, extra: dict | None = None) -> None:
    payload = {
        "arch": {"channels": net.channels, "blocks": net.blocks},
        "state_dict": net.state_dict(),
    }
    if extra:
        payload["extra"] = extra
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_model(path: Path | str) -> AZNet:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    net = AZNet(**payload["arch"])
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net


def load_extra(path: Path | str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload.get("extra", {})
