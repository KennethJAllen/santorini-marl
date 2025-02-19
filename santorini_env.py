import gymnasium as gym
from gymnasium import spaces
import numpy as np

from santorini.game import Game, GameState
from santorini.config import GRID_SIZE, NUM_WORKERS

class SantoriniEnv(gym.Env):
    """
    A Gymnasium environment wrapper for Santorini.
    This example focuses on a single-agent controlling one 'player' in a 2-player game.
    """
    def __init__(self, board_size=GRID_SIZE, num_workers=NUM_WORKERS):
        super().__init__()

        self.board_size = board_size
    
        # Action space: (worker_id, move_index, build_index)
        num_spaces = board_size**2
        self.action_space = spaces.MultiDiscrete([num_workers, num_spaces, num_spaces])

       # Observation space: (row, col, player_id)
        num_players = 2
        obs_low = np.zeros((board_size, board_size, 2), dtype=np.float32)
        obs_high = np.zeros((board_size, board_size, 2), dtype=np.float32)
        # channel 0 (height): 0..4
        obs_high[:,:,0] = 4
        # channel 1 (occupant): 0..2 (where occupant = -1 becomes 0, occupant 0, 1 becomes 1, 2, etc.)
        obs_high[:,:,1] = num_players

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Initialize internal Game
        self.game = Game()
        self.current_player_id = 0  # If controlling just one player, store that ID

    def reset(self, *, seed=None, options=None):
        """
        Resets the game to a new episode.
        Must return (observation, info).
        """
        super().reset(seed=seed)
        self.game.reset()

        # Pick 2-player mode
        self.game.step(2)
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

        # Compute reward
        done = (self.game.get_state() == GameState.GAME_OVER)
        if done:
            # Example: +1 if we are the winner, else 0
            winner = self.game.get_winner()
            if winner and winner.get_player_id() == self.current_player_id:
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
        """Prints the current board state."""
        print("Current board:")
        self.game.get_board().print_board()

    def _get_observation(self):
        """
        Converts the internal game state into a numeric observation 
        that matches self.observation_space.
        """
        obs = self.game.get_observation()
        return obs
