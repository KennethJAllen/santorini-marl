"""Measure engine, net, and MCTS throughput on this machine.

Usage: uv run python -m santorini.az.benchmark [--sims 50] [--games 3]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from santorini.az import fastgame
from santorini.az.config import AZConfig
from santorini.az.mcts import MCTS
from santorini.az.model import AZNet


def bench_random_playouts(seconds: float = 2.0) -> tuple[float, float]:
    """Full random games/sec and plies/sec of the raw engine."""
    rng = np.random.default_rng(0)
    games = plies = 0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        state = fastgame.initial_state()
        while not fastgame.is_terminal(state):
            actions = fastgame.legal_actions(state)
            state = fastgame.step(state, int(rng.choice(actions)))
            plies += 1
        games += 1
    elapsed = time.perf_counter() - start
    return games / elapsed, plies / elapsed


def bench_net_evals(net: AZNet, seconds: float = 2.0) -> float:
    obs = fastgame.canonical_obs(fastgame.initial_state())
    count = 0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        net.predict(obs, setup=True)
        count += 1
    return count / (time.perf_counter() - start)


def bench_selfplay_game(net: AZNet, sims: int, games: int) -> tuple[float, float]:
    """Play full MCTS-vs-itself games; returns (sec/game, sims/sec)."""
    config = AZConfig(sims=sims)
    rng = np.random.default_rng(0)
    total_sims = 0
    start = time.perf_counter()
    for _ in range(games):
        mcts = MCTS(net, config, rng)
        state = fastgame.initial_state()
        while not fastgame.is_terminal(state):
            ids, visits = mcts.run(state, sims, add_noise=True)
            total_sims += sims
            state = fastgame.step(state, int(ids[int(np.argmax(visits))]))
    elapsed = time.perf_counter() - start
    return elapsed / games, total_sims / elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=50)
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--workers", type=int, default=AZConfig().selfplay_workers)
    args = parser.parse_args()

    torch.set_num_threads(1)
    net = AZNet()
    net.eval()

    gps, pps = bench_random_playouts()
    print(f"random playouts: {gps:7.1f} games/s  ({pps:,.0f} plies/s)")

    eps = bench_net_evals(net)
    print(f"net evals (batch 1): {eps:,.0f}/s")

    spg, sps = bench_selfplay_game(net, args.sims, args.games)
    per_worker = 3600 / spg
    print(f"MCTS self-play @ {args.sims} sims: {spg:.2f} s/game ({sps:,.0f} sims/s)")
    print(
        f"est. throughput: {per_worker:,.0f} games/hr/worker, "
        f"{per_worker * args.workers:,.0f} games/hr on {args.workers} workers"
    )


if __name__ == "__main__":
    main()
