"""Ring buffer of compact self-play positions with npz persistence.

Positions are stored as raw state fields plus a sparse visit distribution
(ids + probs over the ~20-64 legal actions); observations are reconstructed
and D4-augmented at sample time, so storage stays ~200 bytes/position and
every epoch sees fresh random symmetries.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np

from santorini.az import fastgame, symmetry
from santorini.az.fastgame import NUM_ACTIONS, State


class Example(NamedTuple):
    heights: np.ndarray  # (5,5) int8
    workers: np.ndarray  # (2,2,2) int8
    placed: int
    player: int
    pi_ids: np.ndarray  # (k,) int16 legal action ids
    pi_probs: np.ndarray  # (k,) float32 normalized visit counts
    z: float  # outcome from this position's player-to-move perspective


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._data: deque[Example] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._data)

    def add(self, examples: Iterable[Example]) -> None:
        self._data.extend(examples)

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """Uniform sample with a random D4 transform per example. Returns
        obs (B,5,5,13), pi (B,1600), mask (B,1600), z (B,), setup (B,)."""
        idxs = rng.integers(len(self._data), size=batch_size)
        transforms = rng.integers(symmetry.NUM_TRANSFORMS, size=batch_size)
        obs = np.empty((batch_size, 5, 5, fastgame.OBS_CHANNELS), dtype=np.float32)
        pi = np.zeros((batch_size, NUM_ACTIONS), dtype=np.float32)
        mask = np.zeros((batch_size, NUM_ACTIONS), dtype=bool)
        z = np.empty(batch_size, dtype=np.float32)
        setup = np.empty(batch_size, dtype=bool)
        for row, (i, t) in enumerate(zip(idxs, transforms)):
            ex = self._data[i]
            state = State(ex.heights, ex.workers, ex.placed, ex.player, -1)
            obs[row] = symmetry.transform_obs(fastgame.canonical_obs(state), t)
            is_setup = ex.placed < fastgame.TOTAL_WORKERS
            ids = symmetry.transform_action_ids(ex.pi_ids, int(t), is_setup)
            pi[row, ids] = ex.pi_probs
            mask[row, ids] = True
            z[row] = ex.z
            setup[row] = is_setup
        return {"obs": obs, "pi": pi, "mask": mask, "z": z, "setup": setup}

    def save(self, path: Path | str) -> None:
        n = len(self._data)
        heights = np.empty((n, 5, 5), dtype=np.int8)
        workers = np.empty((n, 2, 2, 2), dtype=np.int8)
        placed = np.empty(n, dtype=np.uint8)
        player = np.empty(n, dtype=np.uint8)
        z = np.empty(n, dtype=np.float32)
        offsets = np.zeros(n + 1, dtype=np.int64)
        for i, ex in enumerate(self._data):
            heights[i] = ex.heights
            workers[i] = ex.workers
            placed[i] = ex.placed
            player[i] = ex.player
            z[i] = ex.z
            offsets[i + 1] = offsets[i] + len(ex.pi_ids)
        pi_ids = np.empty(offsets[-1], dtype=np.int16)
        pi_probs = np.empty(offsets[-1], dtype=np.float32)
        for i, ex in enumerate(self._data):
            pi_ids[offsets[i] : offsets[i + 1]] = ex.pi_ids
            pi_probs[offsets[i] : offsets[i + 1]] = ex.pi_probs
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            capacity=self.capacity,
            heights=heights,
            workers=workers,
            placed=placed,
            player=player,
            z=z,
            offsets=offsets,
            pi_ids=pi_ids,
            pi_probs=pi_probs,
        )

    @classmethod
    def load(cls, path: Path | str) -> "ReplayBuffer":
        with np.load(path) as data:
            buf = cls(int(data["capacity"]))
            offsets = data["offsets"]
            for i in range(len(data["z"])):
                lo, hi = offsets[i], offsets[i + 1]
                buf._data.append(
                    Example(
                        heights=data["heights"][i],
                        workers=data["workers"][i],
                        placed=int(data["placed"][i]),
                        player=int(data["player"][i]),
                        pi_ids=data["pi_ids"][lo:hi],
                        pi_probs=data["pi_probs"][lo:hi],
                        z=float(data["z"][i]),
                    )
                )
        return buf
