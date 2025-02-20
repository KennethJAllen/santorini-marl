"""Trains the Santorini model."""
from pathlib import Path
from stable_baselines3 import PPO
from santorini_env import SantoriniEnv
from santorini.config import GRID_SIZE

def train(grid_size: int = GRID_SIZE, total_timesteps = 100):
    """Training"""
    # Create environment
    env = SantoriniEnv(grid_size=grid_size)

    # Instantiate the model
    model = PPO("MlpPolicy", env, verbose=1)

    # Training
    model.learn(total_timesteps=total_timesteps)

    # Save the model
    model_path = Path.cwd() / "models" / "santorini_ppo_model"
    model.save(model_path)

if __name__ == "__main__":
    train()
