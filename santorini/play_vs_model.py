"""Play against a trained model in the GUI."""

from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from santorini.game import Game
from santorini.renderer import PygameRenderer
from santorini.selfplay_env import NUM_ACTIONS, NUM_PLAYERS

# Float32 rounding over 1600 actions can trip torch's Simplex() validation
# on healthy logits (see train.py); disable it for inference too.
torch.distributions.Distribution.set_default_validate_args(False)


def play(model_path: Path, human_player: int = 0):
    model = MaskablePPO.load(str(model_path), device="cpu")
    print("Model loaded successfully!")

    game = Game()
    game.step(NUM_PLAYERS)  # select 2-player mode

    renderer = PygameRenderer()

    while not game.is_done():
        renderer.tick(game)

        if game.current_player_idx == human_player:
            # Block until the human clicks a full (move+build) action.
            action = renderer.get_human_action(game)
        else:
            obs = game.board.get_observation(game.current_player_idx)
            mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
            mask[list(game.valid_actions)] = 1
            action = int(
                model.predict(obs, action_masks=mask, deterministic=True)[0]
            )

        game.step(action)

    renderer.tick(game)
    print(f"Game over! Winner is {game.winner}")


def main():
    models = Path(".").glob("models/santorini_*.zip")
    try:
        latest = max(models, key=lambda p: p.stat().st_mtime)
    except ValueError:
        raise RuntimeError("No saved model found in ./models/") from None
    print(f"Loading {latest}")
    play(latest)


if __name__ == "__main__":
    main()
