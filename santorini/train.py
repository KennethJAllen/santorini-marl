"""Frozen-opponent self-play training for Santorini with MaskablePPO.

The learner is a normal single-agent Gym env (`SantoriniSelfPlayEnv`); the
opponent's moves happen inside `env.step()` using a frozen policy sampled from
a snapshot pool that is refreshed with copies of the learner as training
progresses. This replaces the shared-stream PettingZoo/SB3 wrapper approach,
which dropped the losing player's terminal reward and collapsed self-play
into a single cooperative trajectory.
"""

from collections import deque
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from santorini.opponents import (
    GreedyOpponent,
    Opponent,
    OpponentPool,
    RandomOpponent,
)
from santorini.selfplay_env import SantoriniSelfPlayEnv

# Over 1600 actions, float32 softmax row-sums drift past the 1e-6 tolerance
# torch's Simplex() validation demands (~16 rows per million), which crashes
# MaskableCategorical.apply_masking on perfectly healthy logits. Real NaN/inf
# problems are caught loudly by NanGuardCallback instead.
torch.distributions.Distribution.set_default_validate_args(False)


class BoardCNN(BaseFeaturesExtractor):
    """Small conv net over the (5, 5, 11) board observation (HWC layout)."""

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[2]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample.permute(0, 3, 1, 2)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations.permute(0, 3, 1, 2)))


POLICY_KWARGS = dict(
    features_extractor_class=BoardCNN,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=dict(pi=[256], vf=[256]),
)


class NanGuardCallback(BaseCallback):
    """Fail fast with a clear message if the policy weights ever go
    non-finite. This replaces the safety torch's distribution validation
    used to provide (now disabled — it false-alarmed on float32 rounding)."""

    def _on_rollout_end(self) -> None:
        for name, param in self.model.policy.named_parameters():
            if not torch.isfinite(param).all():
                raise RuntimeError(
                    f"Non-finite values in policy parameter '{name}' at "
                    f"{self.num_timesteps:,} steps - the policy has diverged. "
                    "Lower --lr or --target-kl and restart from a snapshot."
                )

    def _on_step(self) -> bool:
        return True


class SnapshotCallback(BaseCallback):
    """Periodically freeze a copy of the learner into the opponent pool."""

    def __init__(
        self,
        pool: OpponentPool,
        pool_dir: Path,
        snapshot_freq: int = 100_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.pool = pool
        self.pool_dir = pool_dir
        self.snapshot_freq = snapshot_freq
        self._last_snapshot = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_snapshot >= self.snapshot_freq:
            self._last_snapshot = self.num_timesteps
            path = self.pool_dir / f"snapshot_{self.num_timesteps:010d}"
            self.model.save(path)
            self.pool.add_snapshot(path.with_suffix(".zip"))
            if self.verbose:
                print(f"[pool] froze snapshot at {self.num_timesteps:,} steps")
        return True


class SelfplayStatsCallback(BaseCallback):
    """Logs learner-perspective self-play stats from episode-end infos:
    win rate (overall and per seat), episode length, and the terminal vs
    shaping reward split. A healthy run has learner_winrate near 0.5 once the
    pool holds recent snapshots; ~1.0 or ~0.0 per seat is a red flag."""

    def __init__(self, window: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self._episodes = deque(maxlen=window)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "learner_won" in info:
                self._episodes.append(info)
        return True

    def _on_rollout_end(self) -> None:
        if not self._episodes:
            return
        eps = list(self._episodes)
        won = np.array([e["learner_won"] for e in eps], dtype=float)
        seats = np.array([e["learner_seat"] for e in eps])
        self.logger.record("selfplay/learner_winrate", float(won.mean()))
        for seat in (0, 1):
            if (seats == seat).any():
                self.logger.record(
                    f"selfplay/learner_winrate_seat{seat}",
                    float(won[seats == seat].mean()),
                )
        self.logger.record(
            "selfplay/ep_len_mean",
            float(np.mean([e["learner_ep_len"] for e in eps])),
        )
        self.logger.record(
            "selfplay/terminal_reward_mean",
            float(np.mean([e["terminal_reward"] for e in eps])),
        )
        self.logger.record(
            "selfplay/shaping_total_mean",
            float(np.mean([e["shaping_total"] for e in eps])),
        )


def play_eval_games(
    model,
    opponent_factory,
    learner_seat: int,
    n_games: int,
    seed: int = 0,
    deterministic: bool = True,
) -> tuple[float, list[int], list[tuple]]:
    """Play `n_games` with the model in `learner_seat` vs a fresh opponent per
    game. Returns (winrate, episode lengths, learner action trajectories)."""
    wins = 0
    lengths: list[int] = []
    trajectories: list[tuple] = []
    for i in range(n_games):
        opponent: Opponent = opponent_factory(np.random.default_rng(seed + i))
        env = SantoriniSelfPlayEnv(opponent=opponent, learner_seat=learner_seat)
        obs, _ = env.reset(seed=seed + i)
        actions = []
        terminated = False
        info = {}
        while not terminated:
            mask = env.action_masks()
            action = int(
                model.predict(obs, action_masks=mask, deterministic=deterministic)[0]
            )
            actions.append(action)
            obs, _reward, terminated, _truncated, info = env.step(action)
        wins += bool(info["learner_won"])
        lengths.append(info["learner_ep_len"])
        trajectories.append(tuple(actions))
    return wins / n_games, lengths, trajectories


class MetricsEvalCallback(BaseCallback):
    """Every `eval_freq` steps, evaluates the learner from both seats against
    random, greedy-heuristic, and latest-snapshot opponents, and logs
    trajectory diversity — the direct detector for the deterministic
    self-play collapse this rewrite fixes."""

    def __init__(
        self,
        pool: OpponentPool,
        eval_freq: int = 25_000,
        n_games: int = 20,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.pool = pool
        self.eval_freq = eval_freq
        self.n_games = n_games
        self._last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self.eval_freq:
            self._last_eval = self.num_timesteps
            self._run_eval()
        return True

    def _run_eval(self) -> None:
        seed = self.num_timesteps
        all_lengths: list[int] = []
        random_trajectories: list[tuple] = []
        opponent_factories = {
            "random": RandomOpponent,
            "greedy": GreedyOpponent,
        }
        for name, factory in opponent_factories.items():
            for seat in (0, 1):
                winrate, lengths, trajs = play_eval_games(
                    self.model, factory, seat, self.n_games, seed=seed
                )
                self.logger.record(f"eval/winrate_vs_{name}_p{seat}", winrate)
                all_lengths.extend(lengths)
                if name == "random":
                    random_trajectories.extend(trajs)
                if self.verbose:
                    print(
                        f"[eval @ {self.num_timesteps:,}] vs {name} as p{seat}: "
                        f"{winrate:.0%}"
                    )

        latest = self.pool.latest
        if latest is not None:
            wr0, lengths0, _ = play_eval_games(
                self.model, lambda rng: latest, 0, self.n_games, seed=seed
            )
            wr1, lengths1, _ = play_eval_games(
                self.model, lambda rng: latest, 1, self.n_games, seed=seed
            )
            self.logger.record(
                "eval/winrate_vs_latest_snapshot", (wr0 + wr1) / 2
            )
            all_lengths.extend(lengths0 + lengths1)

        # Fraction of distinct learner trajectories across vs-random games.
        # Near 1.0 is healthy; near 0 means the policy funnels every game
        # into the same line of play.
        diversity = len(set(random_trajectories)) / len(random_trajectories)
        self.logger.record("eval/traj_diversity", diversity)
        self.logger.record("eval/ep_len_mean", float(np.mean(all_lengths)))
        self.logger.record("eval/ep_len_std", float(np.std(all_lengths)))


def make_env(pool: OpponentPool, shaping_scale: float, gamma: float):
    def _init():
        env = SantoriniSelfPlayEnv(
            opponent_pool=pool, shaping_scale=shaping_scale, gamma=gamma
        )
        return Monitor(env)

    return _init


def train_selfplay(
    model_dir: Path,
    steps: int = 3_000_000,
    seed: int = 0,
    n_envs: int = 8,
    eval_freq: int = 25_000,
    n_eval_games: int = 20,
    snapshot_freq: int = 100_000,
    shaping_scale: float = 0.0,
    gamma: float = 0.99,
    learning_rate: float = 1e-4,
    target_kl: float = 0.03,
    resume: Path | None = None,
) -> Path:
    """Train MaskablePPO against a pool of frozen snapshots of itself."""
    pool = OpponentPool(rng=np.random.default_rng(seed))
    pool_dir = model_dir / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)

    vec_env = DummyVecEnv([make_env(pool, shaping_scale, gamma)] * n_envs)

    tb_log_dir = model_dir / "tb"
    run_name = f"selfplay_{time.strftime('%Y%m%d-%H%M%S')}"

    if resume is not None:
        model = MaskablePPO.load(
            resume,
            env=vec_env,
            device="cpu",
            tensorboard_log=str(tb_log_dir),
            custom_objects={"learning_rate": learning_rate, "target_kl": target_kl},
        )
        # Re-seed the pool with the snapshots already on disk so the learner
        # does not restart against a purely random opponent.
        for snap in sorted(pool_dir.glob("snapshot_*.zip"))[-pool.max_snapshots:]:
            pool.add_snapshot(snap)
        print(
            f"Resumed from {resume} at {model.num_timesteps:,} steps "
            f"({len(pool._snapshots)} pool snapshots re-registered)"
        )
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs=POLICY_KWARGS,
            learning_rate=learning_rate,
            n_steps=256,  # per env; 8 envs -> 2048-step rollout buffer
            batch_size=512,
            n_epochs=4,
            gamma=gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            # Abort an update once the policy drifts this far from the
            # rollout policy; caps how hard any single update can push.
            target_kl=target_kl,
            tensorboard_log=str(tb_log_dir),
            seed=seed,
            verbose=1,
        )

    callbacks = [
        NanGuardCallback(),
        SnapshotCallback(pool, pool_dir, snapshot_freq=snapshot_freq, verbose=1),
        SelfplayStatsCallback(),
        MetricsEvalCallback(pool, eval_freq=eval_freq, n_games=n_eval_games, verbose=1),
    ]

    print(
        f"Training for {steps:,} steps on {n_envs} envs "
        f"(tensorboard: uv run tensorboard --logdir {tb_log_dir})"
    )
    model.learn(
        total_timesteps=steps,
        callback=callbacks,
        tb_log_name=run_name,
        reset_num_timesteps=resume is None,
    )

    save_path = model_dir / f"santorini_selfplay_{time.strftime('%Y%m%d-%H%M%S')}"
    model.save(save_path)
    save_path = save_path.with_suffix(".zip")
    print(f"Model saved to {save_path}")

    vec_env.close()
    return save_path


def final_eval(model_path: Path, n_games: int = 100, seed: int = 10_000) -> None:
    """Post-training report: winrates from both seats vs random and greedy,
    plus trajectory diversity vs random."""
    model = MaskablePPO.load(model_path, device="cpu")
    print(f"\n=== Final evaluation: {model_path.name} ({n_games} games each) ===")
    trajectories: list[tuple] = []
    for name, factory in (("random", RandomOpponent), ("greedy", GreedyOpponent)):
        for seat in (0, 1):
            winrate, lengths, trajs = play_eval_games(
                model, factory, seat, n_games, seed=seed
            )
            if name == "random":
                trajectories.extend(trajs)
            print(
                f"vs {name:<7} as p{seat}: winrate {winrate:.1%}, "
                f"mean len {np.mean(lengths):.1f}"
            )
    diversity = len(set(trajectories)) / len(trajectories)
    print(f"trajectory diversity vs random: {diversity:.1%}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a frozen-opponent self-play MaskablePPO model on Santorini."
    )
    parser.add_argument("--steps", type=int, default=3_000_000,
                        help="Training timesteps (default: 3M); counted on top "
                             "of the checkpoint's steps when using --resume.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0).")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Parallel envs for rollout collection (default: 8).")
    parser.add_argument("--eval-freq", type=int, default=25_000,
                        help="Steps between in-training evals (default: 25k).")
    parser.add_argument("--n-eval-games", type=int, default=20,
                        help="Games per opponent/seat per in-training eval (default: 20).")
    parser.add_argument("--snapshot-freq", type=int, default=100_000,
                        help="Steps between opponent-pool snapshots (default: 100k).")
    parser.add_argument("--shaping-scale", type=float, default=0.0,
                        help="Scale of potential-based height shaping (default: 0 = off).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4).")
    parser.add_argument("--target-kl", type=float, default=0.03,
                        help="Early-stop a PPO update past this approx KL (default: 0.03).")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume training from a saved model/snapshot .zip "
                             "(e.g. models/pool/snapshot_0000400000.zip).")
    parser.add_argument("--final-eval-games", type=int, default=100,
                        help="Games per opponent/seat in the final eval (default: 100).")
    args = parser.parse_args()

    model_dir = Path.cwd() / "models"
    model_dir.mkdir(exist_ok=True)

    save_path = train_selfplay(
        model_dir,
        steps=args.steps,
        seed=args.seed,
        n_envs=args.n_envs,
        eval_freq=args.eval_freq,
        n_eval_games=args.n_eval_games,
        snapshot_freq=args.snapshot_freq,
        shaping_scale=args.shaping_scale,
        learning_rate=args.lr,
        target_kl=args.target_kl,
        resume=args.resume,
    )

    final_eval(save_path, n_games=args.final_eval_games)


if __name__ == "__main__":
    main()
