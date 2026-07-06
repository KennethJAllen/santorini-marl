"""AlphaZero training loop: self-play -> replay buffer -> train candidate ->
arena gate -> eval ladder, with TensorBoard logging and resume.

Usage:
    uv run python -m santorini.az.train --iters 100
    uv run python -m santorini.az.train --resume

Self-play always uses best.pt; a trained candidate replaces it only after
winning >= gate-threshold of the arena games (disable with --no-gating).
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from santorini.az.agents import GreedyAgent, MCTSAgent, RandomAgent, RawPolicyAgent
from santorini.az.arena import play_match
from santorini.az.config import AZConfig
from santorini.az.model import (
    AZNet,
    combined_logits,
    load_model,
    masked_log_softmax,
    obs_to_tensor,
    save_model,
)
from santorini.az.replay import ReplayBuffer
from santorini.az.selfplay import generate_games


def train_candidate(
    net: AZNet,
    buffer: ReplayBuffer,
    config: AZConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Train `net` in place on uniform symmetric samples from the buffer."""
    net.train()
    opt = torch.optim.AdamW(
        net.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    policy_losses, value_losses = [], []
    for _ in range(config.minibatches):
        batch = buffer.sample(config.batch_size, rng)
        obs = obs_to_tensor(batch["obs"])
        pi = torch.from_numpy(batch["pi"])
        mask = torch.from_numpy(batch["mask"])
        setup = torch.from_numpy(batch["setup"])
        z = torch.from_numpy(batch["z"])

        opt.zero_grad()
        move, place, value = net(obs)
        logp = masked_log_softmax(combined_logits(move, place, setup), mask)
        policy_loss = -(pi * logp).sum(dim=-1).mean()
        value_loss = torch.nn.functional.mse_loss(value, z)
        loss = policy_loss + value_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip)
        opt.step()
        policy_losses.append(float(policy_loss))
        value_losses.append(float(value_loss))
    net.eval()
    return {
        "policy": float(np.mean(policy_losses)),
        "value": float(np.mean(value_losses)),
        "total": float(np.mean(policy_losses) + np.mean(value_losses)),
    }


def run_ladder(
    net: AZNet, config: AZConfig, rng: np.random.Generator
) -> dict[str, float]:
    games = config.ladder_games
    mcts_agent = MCTSAgent(net, config.ladder_sims, config, rng)
    results = {
        "mcts_vs_random": play_match(mcts_agent, RandomAgent(rng), games).winrate_a,
        "mcts_vs_greedy": play_match(mcts_agent, GreedyAgent(rng), games).winrate_a,
        # The metric separating net learning from search strength:
        "raw_vs_greedy": play_match(RawPolicyAgent(net), GreedyAgent(rng), games).winrate_a,
    }
    return results


def train(config: AZConfig, iters: int, resume: bool, seed: int = 0) -> None:
    config.models_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(config.tb_dir))
    rng = np.random.default_rng(seed)

    start_iter = 0
    buffer = ReplayBuffer(config.buffer_size)
    if resume and config.state_path.exists():
        state = json.loads(config.state_path.read_text())
        start_iter = state["iteration"] + 1
        if config.buffer_path.exists():
            buffer = ReplayBuffer.load(config.buffer_path)
        best = load_model(config.best_path)
        print(f"resumed at iteration {start_iter} (buffer: {len(buffer)} positions)")
    else:
        torch.manual_seed(seed)
        best = AZNet(config.channels, config.blocks)
        best.eval()
        save_model(best, config.best_path, extra={"iteration": -1})

    for iteration in range(start_iter, start_iter + iters):
        t0 = time.perf_counter()

        # --- self-play with the current best ---
        base_seed = seed * 1_000_003 + iteration * 1_000_000
        examples, stats = generate_games(
            config.best_path, config, config.games_per_iter, base_seed
        )
        buffer.add(examples)
        selfplay_secs = time.perf_counter() - t0
        games_per_hr = config.games_per_iter / (selfplay_secs / 3600)
        writer.add_scalar("selfplay/games_per_hr", games_per_hr, iteration)
        writer.add_scalar(
            "selfplay/mean_plies", np.mean([s.plies for s in stats]), iteration
        )
        writer.add_scalar(
            "selfplay/root_entropy",
            np.mean([s.mean_root_entropy for s in stats]),
            iteration,
        )
        writer.add_scalar(
            "selfplay/seat0_win_frac",
            np.mean([s.seat0_won for s in stats]),
            iteration,
        )
        writer.add_scalar("buffer/size", len(buffer), iteration)

        # --- train a candidate from the current best ---
        candidate = deepcopy(best)
        losses = train_candidate(candidate, buffer, config, rng)
        for name, val in losses.items():
            writer.add_scalar(f"loss/{name}", val, iteration)
        save_model(candidate, config.latest_path, extra={"iteration": iteration})
        save_model(candidate, config.iter_path(iteration), extra={"iteration": iteration})

        # --- arena gate ---
        promoted = True
        arena_winrate = float("nan")
        if config.gating:
            cand_agent = MCTSAgent(
                candidate, config.arena_sims, config, rng,
                sample_moves=config.arena_sample_moves,
            )
            best_agent = MCTSAgent(
                best, config.arena_sims, config, rng,
                sample_moves=config.arena_sample_moves,
            )
            result = play_match(cand_agent, best_agent, config.arena_games)
            arena_winrate = result.winrate_a
            promoted = arena_winrate >= config.gate_threshold
            writer.add_scalar("arena/winrate", arena_winrate, iteration)
            writer.add_scalar("arena/seat0_win_frac", result.seat0_fraction, iteration)
        writer.add_scalar("arena/promoted", int(promoted), iteration)
        if promoted:
            best = candidate
            save_model(best, config.best_path, extra={"iteration": iteration})

        # --- eval ladder ---
        ladder = run_ladder(best, config, rng)
        for name, val in ladder.items():
            writer.add_scalar(f"ladder/{name}", val, iteration)

        # --- persist state ---
        if (iteration + 1) % config.snapshot_every == 0:
            buffer.save(config.buffer_path)
        config.state_path.write_text(
            json.dumps({"iteration": iteration, "seed": seed})
        )
        total_secs = time.perf_counter() - t0
        print(
            f"iter {iteration:3d} | {games_per_hr:6.0f} games/hr | "
            f"loss {losses['total']:.3f} (pi {losses['policy']:.3f} v {losses['value']:.3f}) | "
            f"arena {arena_winrate:.0%}{' PROMOTED' if promoted else ''} | "
            f"greedy {ladder['mcts_vs_greedy']:.0%} raw {ladder['raw_vs_greedy']:.0%} | "
            f"{total_secs:.0f}s"
        )

    buffer.save(config.buffer_path)
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = AZConfig()
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--games-per-iter", type=int, default=defaults.games_per_iter)
    parser.add_argument("--sims", type=int, default=defaults.sims)
    parser.add_argument("--workers", type=int, default=defaults.selfplay_workers)
    parser.add_argument("--minibatches", type=int, default=defaults.minibatches)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--arena-games", type=int, default=defaults.arena_games)
    parser.add_argument("--ladder-games", type=int, default=defaults.ladder_games)
    parser.add_argument("--no-gating", action="store_true")
    parser.add_argument("--models-dir", type=str, default=str(defaults.models_dir))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from pathlib import Path

    config = AZConfig(
        games_per_iter=args.games_per_iter,
        sims=args.sims,
        selfplay_workers=args.workers,
        minibatches=args.minibatches,
        batch_size=args.batch_size,
        lr=args.lr,
        arena_games=args.arena_games,
        arena_sims=args.sims,
        ladder_games=args.ladder_games,
        ladder_sims=args.sims,
        gating=not args.no_gating,
        models_dir=Path(args.models_dir),
    )
    train(config, args.iters, args.resume, args.seed)


if __name__ == "__main__":
    main()
