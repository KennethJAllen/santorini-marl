"""Self-play game generation: single-game generator plus multiprocessing
orchestration.

Workers each load the current best checkpoint once (pool initializer), pin
torch to one thread, and stream completed games back via imap_unordered.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from santorini.az import fastgame
from santorini.az.config import AZConfig
from santorini.az.mcts import MCTS
from santorini.az.model import AZNet, load_model
from santorini.az.replay import Example


@dataclass
class GameStats:
    plies: int
    winner: int
    seat0_won: bool
    mean_root_entropy: float


def play_game(
    net: AZNet, config: AZConfig, rng: np.random.Generator
) -> tuple[list[Example], GameStats]:
    """One self-play game. The training target pi is always the normalized
    pre-temperature visit counts; temperature only affects the move played."""
    mcts = MCTS(net, config, rng)
    state = fastgame.initial_state()
    records: list[tuple[fastgame.State, np.ndarray, np.ndarray]] = []
    entropies: list[float] = []
    ply = 0
    while not fastgame.is_terminal(state):
        ids, visits = mcts.run(state, config.sims, add_noise=True)
        pi = visits.astype(np.float64) / visits.sum()
        records.append((state, ids, pi))
        entropies.append(float(-(pi[pi > 0] * np.log(pi[pi > 0])).sum()))
        if ply < config.temp_moves:
            action = int(rng.choice(ids, p=pi))
        else:
            action = int(ids[int(np.argmax(visits))])
        state = fastgame.step(state, action)
        ply += 1

    winner = state.winner
    examples = [
        Example(
            heights=s.heights,
            workers=s.workers,
            placed=s.placed,
            player=s.player,
            pi_ids=ids.astype(np.int16),
            pi_probs=pi.astype(np.float32),
            z=1.0 if s.player == winner else -1.0,
        )
        for s, ids, pi in records
    ]
    stats = GameStats(
        plies=ply,
        winner=winner,
        seat0_won=winner == 0,
        mean_root_entropy=float(np.mean(entropies)),
    )
    return examples, stats


_worker_net: AZNet | None = None
_worker_config: AZConfig | None = None


def _init_worker(model_path: str, config: AZConfig) -> None:
    global _worker_net, _worker_config
    import torch

    torch.set_num_threads(1)
    _worker_net = load_model(model_path)
    _worker_config = config


def _play_one(seed: int) -> tuple[list[Example], GameStats]:
    return play_game(_worker_net, _worker_config, np.random.default_rng(seed))


def generate_games(
    model_path: Path | str,
    config: AZConfig,
    n_games: int,
    base_seed: int,
    workers: int | None = None,
) -> tuple[list[Example], list[GameStats]]:
    workers = config.selfplay_workers if workers is None else workers
    seeds = [base_seed + i for i in range(n_games)]
    examples: list[Example] = []
    stats: list[GameStats] = []
    if workers <= 1:
        _init_worker(str(model_path), config)
        results = map(_play_one, seeds)
        for exs, st in results:
            examples.extend(exs)
            stats.append(st)
        return examples, stats

    ctx = mp.get_context("forkserver")
    with ctx.Pool(
        workers, initializer=_init_worker, initargs=(str(model_path), config)
    ) as pool:
        for exs, st in pool.imap_unordered(_play_one, seeds, chunksize=1):
            examples.extend(exs)
            stats.append(st)
    return examples, stats
