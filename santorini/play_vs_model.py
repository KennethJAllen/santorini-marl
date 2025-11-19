from pathlib import Path
import supersuit as ss
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from santorini.env import santorini_env
from santorini.renderer import PygameRenderer

# Import both the wrappers and the custom stable policy classes
from santorini.train import (
    SB3ActionMaskWrapper,
    mask_fn,
)


def play(model_path: Path, human_player: int = 0):
    # 1) Create the raw PettingZoo env
    pet_env = santorini_env()

    # 2) Wrap for SB3's invalid‐action masking
    sb3_env = SB3ActionMaskWrapper(pet_env)
    sb3_env.reset()
    mask_env = ActionMasker(sb3_env, mask_fn)

    # 3) Turn it into a single‐env VecEnv
    vec_env = ss.stable_baselines3_vec_env_v0(
        mask_env, num_envs=1, multiprocessing=False
    )

    # 4) Load your trained MaskablePPO
    # Try to load the model - if it's an old model without stable policy, warn the user
    try:
        model = MaskablePPO.load(str(model_path), env=vec_env)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"\n⚠️  Error loading model: {e}")
        print("\nThis might be an old model that was trained with a different policy.")
        print("Please retrain with: python santorini/train.py")
        print("Or delete old models with: rm models/santorini_v1_*.zip")
        raise

    # 5) Set up Pygame renderer and reset
    renderer = PygameRenderer()
    mask_env.reset()
    vec_obs = vec_env.reset()

    # 6) Main loop: human goes when current_player_idx == human_player
    done = False
    while not done:
        # Draw the current Game state
        game = sb3_env.env.game
        renderer.tick(game)

        if game.current_player_idx == human_player:
            # block until the human clicks a full (move+build) action
            action = renderer.get_human_action(game)
        else:
            # let SB3 pick for us (pass the mask in)
            current_mask = mask_env.action_masks()  # from SB3ActionMaskWrapper
            vec_action, _ = model.predict(
                vec_obs, action_masks=[current_mask], deterministic=True
            )
            action = int(vec_action[0])

        # 7) Step both the mask-wrapper and the VecEnv so they're in sync
        _, _, termination, truncation, _ = mask_env.step(action)
        done = termination or truncation
        vec_obs, _, _, _ = vec_env.step([action])

    print("Game over!")


def main():
    models = sorted(Path(".").glob("models/santorini_*.zip"))
    if not models:
        raise RuntimeError("No saved model found in ./models/")
    latest = models[-1]
    print(f"Loading {latest}")
    play(latest)


if __name__ == "__main__":
    main()
