"""Play against a trained model in the GUI."""

import argparse
from pathlib import Path
from typing import Callable

import numpy as np

from santorini.game import Game
from santorini.renderer import PygameRenderer
from santorini.selfplay_env import NUM_ACTIONS, NUM_PLAYERS


def make_ppo_agent(model_path: Path) -> Callable[[Game], int]:
    # Imported lazily so the AZ path does not need SB3.
    import torch
    from sb3_contrib import MaskablePPO

    # Float32 rounding over 1600 actions can trip torch's Simplex() validation
    # on healthy logits (see train.py); disable it for inference too.
    torch.distributions.Distribution.set_default_validate_args(False)

    model = MaskablePPO.load(str(model_path), device="cpu")

    def agent_fn(game: Game) -> int:
        obs = game.board.get_observation(game.current_player_idx)
        mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        mask[list(game.valid_actions)] = 1
        return int(model.predict(obs, action_masks=mask, deterministic=True)[0])

    return agent_fn


def make_az_agent(checkpoint: Path, sims: int = 100) -> Callable[[Game], int]:
    from santorini.az import fastgame
    from santorini.az.agents import MCTSAgent
    from santorini.az.model import load_model

    agent = MCTSAgent(load_model(checkpoint), sims=sims)

    def agent_fn(game: Game) -> int:
        return agent.act(fastgame.from_game(game))

    return agent_fn


def play(agent_fn: Callable[[Game], int], human_player: int = 0):
    game = Game()
    game.step(NUM_PLAYERS)  # select 2-player mode

    renderer = PygameRenderer()

    while not game.is_done():
        renderer.tick(game)

        if game.current_player_idx == human_player:
            # Block until the human clicks a full (move+build) action.
            action = renderer.get_human_action(game)
        else:
            action = agent_fn(game)

        game.step(action)

    renderer.tick(game)
    print(f"Game over! Winner is {game.winner}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--az",
        type=Path,
        default=None,
        help="AlphaZero checkpoint (e.g. models/az/best.pt); plays via MCTS",
    )
    parser.add_argument("--sims", type=int, default=100, help="MCTS sims per move")
    parser.add_argument("--human-player", type=int, default=0, choices=(0, 1))
    args = parser.parse_args()

    if args.az is not None:
        print(f"Loading AZ checkpoint {args.az} ({args.sims} sims/move)")
        agent_fn = make_az_agent(args.az, args.sims)
    else:
        models = Path(".").glob("models/santorini_*.zip")
        try:
            latest = max(models, key=lambda p: p.stat().st_mtime)
        except ValueError:
            raise RuntimeError("No saved model found in ./models/") from None
        print(f"Loading {latest}")
        agent_fn = make_ppo_agent(latest)

    play(agent_fn, human_player=args.human_player)


if __name__ == "__main__":
    main()
