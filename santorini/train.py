"""
Based on https://pettingzoo.farama.org/tutorials/sb3/connect_four/

Uses Stable-Baselines3 to train agents in the Santorini environment using invalid action masking.

For information about invalid action masking in PettingZoo, see https://pettingzoo.farama.org/api/aec/#action-masking
For more information about invalid action masking in SB3, see https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

Original author: Elliot (https://github.com/elliottower)
"""

from pathlib import Path
import time
from typing import Any, Dict, Optional, Type

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import pettingzoo.utils

from santorini.env import santorini_env

# Enable anomaly detection for debugging
torch.autograd.set_detect_anomaly(True)


class StableMaskableCategoricalDistribution(MaskableCategoricalDistribution):
    """
    Numerically stable version of MaskableCategoricalDistribution.
    Applies logit clipping and temperature scaling to prevent numerical underflow.
    """

    def __init__(self, action_dim: int, temperature: float = 1.0, logit_clip: float = 10.0):
        super().__init__(action_dim)
        self.temperature = temperature
        self.logit_clip = logit_clip

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """Create the layer that represents the distribution."""
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(self, action_logits: torch.Tensor, action_masks: Optional[torch.Tensor] = None) -> "StableMaskableCategoricalDistribution":
        """
        Create the distribution given its parameters (action_logits) with numerical stability improvements.
        """
        # Apply logit clipping to prevent extreme values
        action_logits = torch.clamp(action_logits, -self.logit_clip, self.logit_clip)

        # Apply temperature scaling (higher temperature = more exploration)
        action_logits = action_logits / self.temperature

        # Apply action mask by setting invalid actions to very negative values
        if action_masks is not None:
            # Convert to torch tensor if needed
            if isinstance(action_masks, (list, np.ndarray)):
                action_masks = torch.tensor(action_masks, dtype=torch.bool, device=action_logits.device)
            elif not isinstance(action_masks, torch.Tensor):
                action_masks = torch.tensor(action_masks, dtype=torch.bool, device=action_logits.device)
            elif action_masks.dtype != torch.bool:
                action_masks = action_masks.bool()

            # Set masked (invalid) actions to a large negative value
            # Using -1e8 instead of -inf for better numerical stability
            action_logits = torch.where(
                action_masks,
                action_logits,
                torch.tensor(-1e8, dtype=action_logits.dtype, device=action_logits.device)
            )

        self.distribution = torch.distributions.Categorical(logits=action_logits)
        return self

    def apply_masking(self, action_masks: torch.Tensor) -> None:
        """
        Apply masking to the distribution (alternative interface for compatibility).
        """
        # Get current logits
        action_logits = self.distribution.logits

        # Convert to torch tensor if needed
        if isinstance(action_masks, (list, np.ndarray)):
            action_masks = torch.tensor(action_masks, dtype=torch.bool, device=action_logits.device)
        elif not isinstance(action_masks, torch.Tensor):
            action_masks = torch.tensor(action_masks, dtype=torch.bool, device=action_logits.device)
        elif action_masks.dtype != torch.bool:
            action_masks = action_masks.bool()

        # Apply mask
        action_logits = torch.where(
            action_masks,
            action_logits,
            torch.tensor(-1e8, dtype=action_logits.dtype, device=action_logits.device)
        )

        # Recreate distribution with masked logits
        self.distribution = torch.distributions.Categorical(logits=action_logits)


class StableMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    """
    Custom policy that uses the stable maskable categorical distribution.
    """

    def __init__(self, *args, temperature: float = 1.5, logit_clip: float = 10.0, **kwargs):
        self.temperature = temperature
        self.logit_clip = logit_clip
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        """
        Create the networks and the optimizer.
        Override to use our stable distribution.
        """
        super()._build(lr_schedule)

        # Replace the action distribution with our stable version
        self.action_dist = StableMaskableCategoricalDistribution(
            self.action_space.n,
            temperature=self.temperature,
            logit_clip=self.logit_clip
        )


class DebugCallback(BaseCallback):
    """Callback to debug action masking during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.step_count = 0
        self.last_entropy_loss = None

    def _on_step(self):
        self.step_count += 1
        # Access the current observation and action mask
        if hasattr(self.training_env, 'get_attr'):
            try:
                masks = self.training_env.env_method('action_mask')
                obs = self.locals.get('obs_tensor', None)

                if masks:
                    mask = masks[0]
                    num_valid = np.sum(mask)

                    # Print every 1000 steps or when there are very few valid actions
                    if self.step_count % 1000 == 0:
                        status_msg = f"Step {self.step_count}: Valid actions: {num_valid}/{len(mask)}"
                        if self.last_entropy_loss is not None:
                            status_msg += f", Entropy loss: {self.last_entropy_loss:.3f}"
                        print(status_msg)

                    # if num_valid < 10:
                    #     print(f"  Warning: Very few valid actions! Mask: {np.where(mask)[0]}")

            except Exception as e:
                if self.step_count % 1000 == 0:
                    print(f"Step {self.step_count}: Could not access mask - {e}")

        return True

    def _on_rollout_end(self):
        """Called at the end of each rollout to track entropy."""
        # Try to get the entropy loss from the logger
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            try:
                # Get the last recorded entropy loss
                if 'train/entropy_loss' in self.model.logger.name_to_value:
                    self.last_entropy_loss = self.model.logger.name_to_value['train/entropy_loss']

                    # Warn if entropy is getting too negative (policy too deterministic)
                    # For sparse action spaces (8-50 valid actions out of 1600), entropy around -2 to -4 is expected
                    if self.last_entropy_loss < -4.5:
                        print(f"  ⚠️  WARNING: Entropy loss is very negative ({self.last_entropy_loss:.3f}) - policy may be too deterministic!")
                    elif self.step_count % 5000 == 0:
                        print(f"  Entropy loss: {self.last_entropy_loss:.3f}")
            except Exception:
                pass


# To pass into other gymnasium wrappers, we need to ensure that pettingzoo's wrappper
# can also be a gymnasium Env. Thus, we subclass under gym.Env as well.
class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper, gym.Env):
    """Wrapper to allow PettingZoo environments to be used with SB3 illegal action masking."""

    def reset(self, seed=None, options=None):
        """Gymnasium-like reset function which assigns obs/action spaces to be the same for each agent.

        This is required as SB3 is designed for single-agent RL and doesn't expect obs/action spaces to be functions
        """
        super().reset(seed, options)

        # Strip the action mask out from the observation space
        self.observation_space = super().observation_space(self.possible_agents[0])[
            "observation"
        ]
        self.action_space = super().action_space(self.possible_agents[0])

        # Return initial observation, info (PettingZoo AEC envs do not by default)
        return self.observe(self.agent_selection), {}

    def step(self, action):
        """
        Gymnasium-like step function, returning observation, reward, termination, truncation, info.
        The observation is for the next agent (used to determine the next action), while the remaining
        items are for the agent that just acted (used to understand what just happened).
        """
        current_agent = self.agent_selection

        super().step(action)

        next_agent = self.agent_selection
        return (
            self.observe(next_agent),
            self._cumulative_rewards[current_agent],
            self.terminations[current_agent],
            self.truncations[current_agent],
            self.infos[current_agent],
        )

    def observe(self, agent):
        """Return only raw observation, removing action mask."""
        return super().observe(agent)["observation"]

    def action_mask(self):
        """Separate function used in order to access the action mask."""
        return super().observe(self.agent_selection)["action_mask"]


def mask_fn(env):
    mask = env.action_mask()
    num_valid = np.sum(mask)
    total_actions = len(mask)

    # Print diagnostic information periodically or when there are few valid actions
    # if num_valid < 10:
    #     print(f"\n=== MASK WARNING ===")
    #     print(f"Valid actions: {num_valid}/{total_actions}")
    #     print(f"Valid action indices: {np.where(mask)[0]}")
    #     print(f"Current agent: {env.unwrapped.agent_selection}")
    #     print(f"===================\n")

    # Ensure there is at least one valid action
    assert any(mask), f"No valid actions! Agent: {env.unwrapped.agent_selection}"

    return mask


def train_action_mask(env_fn, model_dir: Path, steps=10_000, seed=0,
                      temperature=1.5, logit_clip=10.0, **env_kwargs):
    """
    Train a single model to play as each agent in a zero-sum game environment using invalid action masking.

    Args:
        temperature: Temperature for softmax (higher = more exploration, default 1.5)
        logit_clip: Clip logits to [-logit_clip, logit_clip] for numerical stability (default 10.0)
    """
    env = env_fn(**env_kwargs)

    print(f"Starting training on {str(env.metadata['name'])}.")
    print(f"Using temperature={temperature}, logit_clip={logit_clip} for numerical stability")

    # Custom wrapper to convert PettingZoo envs to work with SB3 action masking
    env = SB3ActionMaskWrapper(env)

    env.reset(seed=seed)  # Must call reset() in order to re-define the spaces

    env = ActionMasker(env, mask_fn)  # Wrap to enable masking (SB3 function)

    # Create policy kwargs with temperature and logit clipping
    policy_kwargs = {
        "temperature": temperature,
        "logit_clip": logit_clip,
    }

    # MaskablePPO with our numerically stable policy
    model = MaskablePPO(
        StableMaskableActorCriticPolicy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,      # Reduced from 3e-4 for more stable training
        n_steps=2048,            # Collect more steps before updating (helps with sparse rewards)
        batch_size=512,          # Larger batch size for stability
        n_epochs=10,             # Number of optimization epochs per update
        gamma=0.99,              # Discount factor (important for long games)
        gae_lambda=0.95,         # Generalized Advantage Estimation parameter
        clip_range=0.2,          # PPO clipping parameter
        ent_coef=0.2,            # Increased to 0.2 to maintain exploration with sparse action space
        vf_coef=0.5,             # Value function coefficient
        max_grad_norm=0.5,       # Gradient clipping
        verbose=1
    )
    model.set_random_seed(seed)

    # Train with debugging callback and error handling
    try:
        print("Starting training with debugging enabled...")
        model.learn(total_timesteps=steps, callback=DebugCallback())
    except ValueError as e:
        print("\n=== ERROR OCCURRED DURING TRAINING ===")
        print(f"Error: {e}")
        print(f"\nCurrent environment state:")
        try:
            print(f"  Agent: {env.unwrapped.agent_selection}")
            current_mask = env.action_mask()
            print(f"  Action mask: {current_mask}")
            print(f"  Num valid actions: {np.sum(current_mask)}")
            print(f"  Valid action indices: {np.where(current_mask)[0]}")

            # Try to get the current observation
            obs = env.observe(env.unwrapped.agent_selection)
            print(f"  Observation shape: {obs.shape}")
            print(f"  Observation stats: min={obs.min()}, max={obs.max()}, mean={obs.mean()}")
        except Exception as inner_e:
            print(f"  Could not retrieve environment state: {inner_e}")

        print("=" * 40)
        raise

    save_path = model_dir / f"{env.unwrapped.metadata['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    model.save(save_path)

    print(f"Model has been saved to {save_path}")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

    env.close()


def eval_action_mask(env_fn, model_dir: Path, num_games: int=100, render_mode:str=None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn(render_mode=render_mode, **env_kwargs)

    print(
        f"Starting evaluation vs a random agent. Trained agent will play as {env.possible_agents[1]}."
    )

    try:
        env_name = env.metadata["name"]
        latest_policy = max(
            model_dir.glob(f"{env_name}*.zip"),
            key=lambda p: p.stat().st_ctime,   # creation time
            )
    except ValueError:
        print("Policy not found.")
        return

    model = MaskablePPO.load(latest_policy)

    scores = {agent: 0 for agent in env.possible_agents}
    total_rewards = {agent: 0 for agent in env.possible_agents}
    round_rewards = []

    for i in range(num_games):
        env.reset(seed=i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            # Separate observation and action mask
            observation, action_mask = obs.values()

            if termination or truncation:
                # If there is a winner, keep track, otherwise don't change the scores (tie)
                if (
                    env.rewards[env.possible_agents[0]]
                    != env.rewards[env.possible_agents[1]]
                ):
                    winner = max(env.rewards, key=env.rewards.get)
                    scores[winner] += env.rewards[
                        winner
                    ]  # only tracks the largest reward (winner of game)
                # Also track negative and positive rewards (penalizes illegal moves)
                for a in env.possible_agents:
                    total_rewards[a] += env.rewards[a]
                # List of rewards by round, for reference
                round_rewards.append(env.rewards)
                break
            else:
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample(action_mask)
                else:
                    # Note: PettingZoo expects integer actions
                    act = int(model.predict(observation, action_masks=action_mask, deterministic=True)[0])
            env.step(act)
            if render_mode == "rgb_array":
                env.render()
                time.sleep(0.5)
    env.close()

    # Avoid dividing by zero
    if sum(scores.values()) == 0:
        winrate = 0
    else:
        winrate = scores[env.possible_agents[1]] / sum(scores.values())
    print("Rewards by round: ", round_rewards)
    print("Total rewards (incl. negative rewards): ", total_rewards)
    print("Winrate: ", winrate)
    print("Final scores: ", scores)
    return round_rewards, total_rewards, winrate, scores

def main():
    env_fn = santorini_env
    env_kwargs = {}
    model_dir = Path.cwd() / "models"
    model_dir.mkdir(exist_ok=True)

    # Train a model against itself
    num_steps = 1_000_000

    train_action_mask(env_fn, model_dir, steps=num_steps, seed=0, **env_kwargs)

    # Evaluate 1000 games against a random agent
    eval_action_mask(env_fn, model_dir, num_games=500, render_mode=None, **env_kwargs)

    # Watch two games vs a random agent
    eval_action_mask(env_fn, model_dir, num_games=2, render_mode="rgb_array", **env_kwargs)


if __name__ == "__main__":
    main()
