from pathlib import Path
from stable_baselines3 import PPO
from santorini.santorini_env import SantoriniEnv

def main():
    model_path = Path.cwd() / "models" / "santorini_ppo_model"
    model = PPO.load(model_path)
    env = SantoriniEnv()
    obs, info = env.reset()

    done = False
    while not done:
        # 1. The model predicts an action given the current observation
        action, _states = model.predict(obs)
        # 2. Step the env
        obs, reward, done, truncated, info = env.step(action)
        # 3. (Optional) Render or print the board
        env.render()

    print("Episode reward:", reward)

if __name__ == "__main__":
    main()
