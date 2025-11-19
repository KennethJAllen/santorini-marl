"""PettingZoo Santorini environment for multi-agent reinforcment learning training."""
import numpy as np
import gymnasium
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import AgentSelector
from santorini.game import Game, GameState
from santorini.renderer import PygameRenderer

def santorini_env(**kwargs):
    env = SantoriniEnv(**kwargs)
    # enforce the AEC API calls
    env = wrappers.OrderEnforcingWrapper(env)
    # catch out-of-bounds ints
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # handle in-bounds but "illegal" game moves
    # Reduced penalty from -1 to -0.1 to prevent policy from becoming too conservative
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-0.1)
    return env

class SantoriniEnv(AECEnv):
    """
    A 2-player turn-based Santorini environment using PettingZoo's AEC API.
    """
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "name": "santorini_v1",
        "is_parallelizable": False,
        "render_fps": 2
    }

    def __init__(self, num_players: int = 2, render_mode: str = None):
        super().__init__()
        # Initialize internal Game
        self.game = Game()
        self.num_players = num_players

        self.agents = [f"player_{i}" for i in range(num_players)]
        self.possible_agents = self.agents[:]
        self.agent_to_idx = {agent: idx for idx, agent in enumerate(self.possible_agents)}
        self._agent_selector = AgentSelector(self.agents)

        self.action_spaces = {name: spaces.Discrete(5*5*8*8) for name in self.agents}
        self.observation_spaces = {
                    name: spaces.Dict(
                        {
                            "observation": spaces.Box(low=0, high=1, shape=(5, 5, 11), dtype=np.int8),
                            "action_mask": spaces.Box(low=0, high=1, shape=(5*5*8*8,), dtype=np.int8),
                            }
                        )
                    for name in self.agents
                    }

        self.rewards = None
        self.infos = {name: {} for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.agent_selection = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if render_mode in ("human", "rgb_array"):
            self.renderer = PygameRenderer(
                grid_size=5,
                asset_dir="images/assets",
                screen_size=600
            )

    def reset(self, seed=None, options=None):
        """Resets the game"""
        self.agents = self.possible_agents[:]
        self.game.reset()
        # Pick 2-player mode
        self.game.step(self.num_players)
        # Now the game is in SETUP state, waiting for worker placements.
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {name: 0 for name in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        current_index = self.agent_to_idx[agent]
        observation = self.game.board.get_observation(current_index)

        legal_moves = self.game.valid_actions if agent == self.agent_selection else set()

        action_mask = np.zeros(5 * 5 * 8 * 8, "int8")
        for i in legal_moves:
            action_mask[i] = 1

        return {"observation": observation, "action_mask": action_mask}

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return self._was_dead_step(action)

        # Reset rewards for current agent before calculating new reward
        self.rewards[self.agent_selection] = 0

        self.game.step(action)

        game_over = self.game.is_done()

        if game_over:
            # Set rewards for winning
            result_val = 1 if self.game.winner == self.game.players[0] else -1
            self.set_game_result(result_val)
        elif self.game.state == GameState.PLAYING:
            # Strategic reward during gameplay
            player_idx = self.agent_to_idx[self.agent_selection]
            opp_idx = 1 - player_idx
            reward = self._calculate_strategic_reward(player_idx, opp_idx)
            self.rewards[self.agent_selection] = reward
        elif self.game.state == GameState.SETUP:
            # Setup phase reward shaping
            player_idx = self.agent_to_idx[self.agent_selection]
            reward = self._calculate_setup_reward(player_idx)
            self.rewards[self.agent_selection] = reward

        self._accumulate_rewards()

        # Give turn to the next agent
        self.agent_selection = self._agent_selector.next()

    def _calculate_strategic_reward(self, player_idx: int, opp_idx: int) -> float:
        """
        Calculate a strategic reward that combines multiple signals:
        1. Maximum height advantage (ability to win soon)
        2. Average height advantage (overall board control)
        3. Mobility advantage (number of valid moves)
        4. Win threat bonus (can reach height 3 next turn)
        """
        player = self.game.players[player_idx]
        opp = self.game.players[opp_idx]

        # Safety check: ensure workers are placed
        if not player.workers or not opp.workers:
            return 0.0

        # Maximum height (most important - closer to winning)
        player_max_height = max(self.game.board.get_height(w.position) for w in player.workers if w.position is not None)
        opp_max_height = max(self.game.board.get_height(w.position) for w in opp.workers if w.position is not None)
        max_height_reward = 0.3 * (player_max_height - opp_max_height)

        # Average height (general board control)
        player_avg_height = sum(self.game.board.get_height(w.position) for w in player.workers if w.position is not None) / len(player.workers)
        opp_avg_height = sum(self.game.board.get_height(w.position) for w in opp.workers if w.position is not None) / len(opp.workers)
        avg_height_reward = 0.1 * (player_avg_height - opp_avg_height)

        # Mobility (number of valid actions)
        player_valid_actions = set()
        for worker in player.workers:
            player_valid_actions |= self.game.board.get_valid_worker_actions(worker)

        opp_valid_actions = set()
        for worker in opp.workers:
            opp_valid_actions |= self.game.board.get_valid_worker_actions(worker)

        # Normalize by typical number of moves (~20-40)
        mobility_reward = 0.01 * (len(player_valid_actions) - len(opp_valid_actions))

        # Win threat bonus: can any worker reach height 3?
        player_can_win = False
        for worker in player.workers:
            if worker.position is not None:
                worker_position = worker.position
                valid_moves = self.game.board._get_valid_moves_from_position(worker_position)
                if any(self.game.board.get_height(move_pos) == 3 for move_pos in valid_moves):
                    player_can_win = True
                    break

        opp_can_win = False
        for worker in opp.workers:
            if worker.position is not None:
                worker_position = worker.position
                valid_moves = self.game.board._get_valid_moves_from_position(worker_position)
                if any(self.game.board.get_height(move_pos) == 3 for move_pos in valid_moves):
                    opp_can_win = True
                    break

        win_threat_reward = 0.2 if player_can_win else 0.0
        win_threat_reward -= 0.2 if opp_can_win else 0.0

        return max_height_reward + avg_height_reward + mobility_reward + win_threat_reward

    def _calculate_setup_reward(self, player_idx: int) -> float:
        """
        Calculate reward during setup phase.
        Encourages:
        - Placing workers near center (more mobility)
        - Keeping workers separated (more board coverage)
        - Avoiding corners (less mobility)
        """
        player = self.game.players[player_idx]
        if not player.workers:
            return 0.0

        reward = 0.0
        last_worker = player.workers[-1]  # The worker just placed
        x, y = last_worker.position

        # Center bonus: reward positions closer to center
        center = self.game.board.grid_size / 2
        distance_from_center = abs(x - center) + abs(y - center)
        max_distance = 2 * (center - 0.5)  # Maximum Manhattan distance from center
        center_reward = 0.05 * (1 - distance_from_center / max_distance)
        reward += center_reward

        # Separation bonus: penalize workers too close together
        if len(player.workers) > 1:
            for other_worker in player.workers[:-1]:
                ox, oy = other_worker.position
                distance = max(abs(x - ox), abs(y - oy))  # Chebyshev distance
                if distance <= 1:
                    reward -= 0.1  # Penalty for adjacent workers
                elif distance == 2:
                    reward -= 0.05  # Small penalty for very close workers

        return reward

    def render(self):
        """
        Renders the game depending on the render mode.
        ansi for string based board representation
        human or rgb_array for gui
        """
        if self.render_mode is None:
            gymnasium.logger.warn("You are calling render method without specifying any render mode.")
        elif self.render_mode == "ansi":
            return str(self.game.board)
        elif self.render_mode in ("human", "rgb_array"):
            # draw board, workers, and highlights
            self.renderer.draw(self.game)

            # let the user click once (or three times, depending on phase),
            # and return the raw action int as soon as we have one
            if self.render_mode == "human":
                return self.renderer.get_human_action(self.game)
            else:
                return None
        else:
            raise ValueError(
                (f"{self.render_mode} is not a valid render mode. "
                 f"Available modes are: {self.metadata['render_modes']}")
            )

    def set_game_result(self, result_val):
        for i, name in enumerate(self.agents):
            self.terminations[name] = True
            result_coef = 1 if i == 0 else -1
            reward = result_val * result_coef
            self.rewards[name] = reward
            self.infos[name] = {"legal_moves": []}

def main():
    # make an env with human rendering
    env = santorini_env(render_mode="human")
    env.reset()

    # game loop
    done = False
    while not done:
        action = env.render()
        if action is not None:
            env.step(action)
            done = all(env.terminations.values())

    print(f"Game over! Winner is {env.game.winner}")
    env.close()

if __name__ == "__main__":
    main()
