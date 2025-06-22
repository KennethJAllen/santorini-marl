"""PettingZoo Santorini environment for multi-agent reinforcment learning training."""
import numpy as np
import gymnasium
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import AgentSelector
from santorini.game import Game
from santorini.renderer import PygameRenderer

def santorini_env(**kwargs):
    env = SantoriniEnv(**kwargs)
    # enforce the AEC API calls
    env = wrappers.OrderEnforcingWrapper(env)
    # catch out-of-bounds ints
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # handle in-bounds but “illegal” game moves
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
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
                            "observation": spaces.Box(low=0, high=1, shape=(5, 5, 7), dtype=np.int8),
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

        self.game.step(action)

        game_over = self.game.is_done()

        if game_over:
            # Set rewards for winning
            result_val = 1 if self.game.winner == self.game.players[0] else -1
            self.set_game_result(result_val)
        else:
            player_idx = self.agent_to_idx[self.agent_selection]
            opp_idx = 1 - player_idx
            player_heights = sum(self.game.board.get_height(worker.position) for worker in self.game.players[player_idx].workers)
            opp_heights = sum(self.game.board.get_height(worker.position) for worker in self.game.players[opp_idx].workers)
            self.rewards[self.agent_selection] = 0.1 * (player_heights - opp_heights)

        self._accumulate_rewards()

        # Give turn to the next agent
        self.agent_selection = self._agent_selector.next()

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
