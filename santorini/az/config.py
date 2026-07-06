"""All AlphaZero hyperparameters and paths in one dataclass."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_workers() -> int:
    return max(1, (os.cpu_count() or 3) - 2)


@dataclass
class AZConfig:
    # network
    channels: int = 64
    blocks: int = 4

    # MCTS
    sims: int = 50
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    # plies played at temperature 1 before switching to argmax; 12 covers all
    # 4 placements plus the first few moves.
    temp_moves: int = 12

    # self-play
    games_per_iter: int = 500
    selfplay_workers: int = field(default_factory=_default_workers)

    # replay buffer
    buffer_size: int = 400_000
    snapshot_every: int = 5  # persist buffer every N iterations

    # optimization
    minibatches: int = 1000
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # arena gating (candidate vs best)
    gating: bool = True
    arena_games: int = 40  # total; half per seat
    arena_sims: int = 50
    gate_threshold: float = 0.55
    # each agent samples its first N own moves from the visit distribution so
    # net-vs-net games differ from one another (deterministic play would
    # collapse the match to one distinct game per seat)
    arena_sample_moves: int = 4

    # eval ladder (best vs random/greedy)
    ladder_games: int = 40  # total per opponent; half per seat
    ladder_sims: int = 50

    # paths
    models_dir: Path = Path("models/az")

    @property
    def tb_dir(self) -> Path:
        return self.models_dir / "tb"

    @property
    def best_path(self) -> Path:
        return self.models_dir / "best.pt"

    @property
    def latest_path(self) -> Path:
        return self.models_dir / "latest.pt"

    @property
    def state_path(self) -> Path:
        return self.models_dir / "state.json"

    @property
    def buffer_path(self) -> Path:
        return self.models_dir / "buffer.npz"

    def iter_path(self, iteration: int) -> Path:
        return self.models_dir / f"iter_{iteration:03d}.pt"
