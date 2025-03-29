"""Santorini environment for multi-agent reinforcment learning training."""
import numpy as np
import gymnasium
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector

from santorini.game import Game, GameState

class SantoriniEnv(AECEnv):
    """
    A 2-player turn-based Santorini environment using PettingZoo's AEC API.
    """
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "name": "santorini_v0",
        "render_fps": 2
    }

    def __init__(self, num_players: int = 2, render_mode: str = None):
        super().__init__()
        # Initialize internal Game
        self.game = Game()
        self._num_players = num_players

        self.agents = [f"player_{i}" for i in range(num_players)]
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        """Resets the game"""
        super().reset(seed=seed)
        self.game.reset()

        # Pick 2-player mode
        self.game.step(self._num_players)
        # Now the game is in SETUP state, waiting for worker placements.

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        """
        Receives a MultiDiscrete action of length 3.
        Interprets it based on the current game state.
        return (observation, reward, done, truncated, info).
        """
        state = self.game.get_state()
        if state == GameState.SETUP:
            # The game expects a single integer in [0..24] for a 5x5 board.
            # We'll interpret the second argument as that position.
            _, pos_index, _ = action
            self.game.step(pos_index)

        elif state == GameState.PLAYING:
            # The game expects a tuple (worker_id, move_index, build_index).
            try:
                self.game.step(action)
            except ValueError:
                # If the action is invalid
                pass

        done = self.game.is_done()
        if done:
            # Example: +1 if we are the winner, else 0
            winner = self.game.get_winner()
            if winner and winner.get_id() == self.current_player_id:
                reward = 1.0
            else:
                reward = -1.0
        else:
            reward = 0.0

        # Build next observation
        observation = self._get_observation()

        truncated = False
        info = {}

        return observation, reward, done, truncated, info

    def render(self):
        """
        Renders the game depending on the render mode.
        ansi for string based board representation
        human or rgb_array for gui
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
        elif self.render_mode == "ansi":
            return str(self.game.get_board())
        elif self.render_mode in {"human", "rgb_array"}:
            raise NotImplementedError
        else:
            raise ValueError(
                f"{self.render_mode} is not a valid render mode. Available modes are: {self.metadata['render_modes']}"
            )

    def action_space(self, agent):
        num_spaces = self._grid_size**2
        return spaces.MultiDiscrete([self._num_workers, num_spaces, num_spaces])

    def observation_space(self, agent):
       # Observation space: (row, col, player_id)
        obs_low = np.zeros((self._grid_size, self._grid_size, 3), dtype=np.float32)
        obs_high = np.zeros((self._grid_size, self._grid_size, 3), dtype=np.float32)
        # channel 0 (height): 0..4
        obs_high[:,:,0] = 4
        # channel 1 (occupant): 0 empty, 1 first player, 2 second player
        obs_high[:,:,1] = self._num_players
        # channel 2 (turn player): 0 first player's turn, 1 second player's turn
        obs_high[:,:,2] = self._num_players

        return spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def _get_observation(self):
        """
        Converts the internal game state into a numeric observation 
        that matches self.observation_space.
        """
        obs = self.game.get_observation()
        return obs
