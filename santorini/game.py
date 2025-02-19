"""Main Santorini game state logic"""
import enum
import numpy as np
from santorini.board import Board
from santorini.player import Player, Worker
from santorini import utils
from santorini.config import NUM_WORKERS

class GameState(enum.Enum):
    """Encodes finite game states."""
    PLAYER_SELECT = 1
    SETUP = 2
    PLAYING = 3
    GAME_OVER = 4

class Game:
    """Game logic, setup, and main loop."""
    def __init__(self):
        self._board = Board() # The game board, an instance of the Board class
        self._players: list[Player] = [] # List of Player objects participating in the game
        self._current_player_index: int = 0  # Index to keep track of whose turn it is
        self._state = GameState.PLAYER_SELECT
        self._winner = None # the winner of the game

    def reset(self) -> None:
        """Sets the board back to start."""
        self._board = Board()
        self._players = []
        self._current_player_index = 0
        self._state = GameState.PLAYER_SELECT
        self._winner = None # the winner of the game

    def step(self, action) -> None:
        """
        Updates the game with the given action.
        When in the player select phase, the action is an integer representing the number of players in the game.
        When in the setup phase, the action represents a location to place a piece.
        When in the playing phase, 
        """
        if self._state == GameState.PLAYER_SELECT:
            self._handle_player_select(action)
        elif self._state == GameState.SETUP:
            self._handle_setup(action)
        elif self._state == GameState.PLAYING:
            self._handle_turn(action)
        elif self._state == GameState.GAME_OVER:
            pass
        else:
            raise ValueError(f"State not handles by 'update_game': {self._state}")

    def get_observation(self) -> np.ndarray:
        """
        Returns an array-based representation of the board state
        shape=(5,5,2)
        channel 1: building height
        channel 2: which player (first, second, third) occupies each cell (or 0 if empty)
        """
        return self._board.get_observation()

    def is_done(self) -> bool:
        """True if game is over, false otherwise."""
        return self._state == GameState.GAME_OVER

    def _handle_player_select(self, action: int) -> None:
        """Action is the number of players chosen."""
        valid_actions = {2, 3}
        if action not in valid_actions:
            raise ValueError(f"Number of players must be 2 or 3. Instead got: {action}")
        num_players = action
        self._init_players(num_players)
        self._state = GameState.SETUP

    def _handle_setup(self, action) -> None:
        """
        Sets player pieces on the board.
        Takes an action which is a position index from 0 to 25.
        """
        current_player = self.get_current_player()
        valid_actions = self._board.get_valid_placement_actions()
        if not action in valid_actions:
            raise ValueError(f"Invalid action: {action}")

        position = utils.space_index_to_position(action)
        worker_id = len(current_player.get_workers())
        new_worker = Worker(worker_id=worker_id, player=current_player)
        current_player.add_worker(new_worker)
        self._board.place_worker(position, new_worker)

        if len(current_player.get_workers()) >= NUM_WORKERS:
            self._current_player_index += 1
            if self._current_player_index >= len(self._players):
                self._current_player_index = 0
                self._update_all_valid_actions()
                self._state = GameState.PLAYING

    def _handle_turn(self, action: tuple[int, int, int]) -> None:
        """
        Applies the action in the form of (worker_id, move_index, build_index)
        to the game state if it is a valid action.
        """
        current_player = self._players[self._current_player_index]
        if action not in current_player.get_valid_actions():
            raise ValueError(f"Invalid action: {action}")

        worker_id, move_index, build_index = action
        worker = current_player.get_worker(worker_id)
        move_position = utils.space_index_to_position(move_index)
        self._board.move_worker(worker, move_position)

        build_position = utils.space_index_to_position(build_index)
        self._board.build(build_position)
        if self._board.is_done():
            self._state = GameState.GAME_OVER
            self._winner = current_player
        else:
            self._update_player_valid_actions(current_player)
            self._cycle_turn()

    def _cycle_turn(self) -> None:
        """Passes the turn to the next player and checks if they have a valid move."""
        current_player = self.get_current_player()
        # cycle through player turns
        self._current_player_index = (self._current_player_index + 1) % len(self._players)
        next_player = self.get_current_player()

        # check that next player has a valid move
        if not next_player.get_valid_actions():
            # TODO: fix this logic for 3 players.
            # If a player has no valid moves, their pieces should be removed from the game.
            self._winner = current_player
            self._state = GameState.GAME_OVER

    def _init_players(self, num_players) -> None:
        """Initializes the players in the game."""
        for player_id in range(num_players):
            self._players.append(Player(player_id))

    def _update_player_valid_actions(self, player: Player) -> None:
        """
        Updates all valid actions the player can take.
        An action is of the form (worker_id, move_index, build_index).
        If a move will win the game (by moving a piece to a height 3 building), the build index is arbitrary.
        """
        player_valid_actions = set()
        for worker in player.get_workers():
            worker_valid_actions = self._board.get_valid_worker_actions(worker)
            # Add worker's valid actions to total set of player's valid actions
            player_valid_actions = player_valid_actions | worker_valid_actions
        player.set_valid_actions(player_valid_actions)

    def _update_all_valid_actions(self) -> None:
        """
        Updates the valid actions for all players.
        An action is of the form (worker_id, move_index, build_index).
        """
        for player in self._players:
            self._update_player_valid_actions(player)

    def get_current_player(self) -> Player:
        """Returns the current player."""
        return self._players[self._current_player_index]

    def get_state(self) -> GameState:
        """Returns the game state"""
        return self._state

    def get_board(self) -> Board:
        """Returns the game board"""
        return self._board

    def get_winner(self) -> Player | None:
        """Returns the winner if the game if one exists. Else return None."""
        return self._winner

    # def ai_take_turn(self, ai_model):
    #     obs = self._encode_observation_for_ai()
    #     action = ai_model.predict(obs)  # or a similar call
#     2#     self.apply_action(action)

# def main():
#     """Command line interface for playing the game."""
#     game = Game()
#     # initi 2 players
#     game.step(2)
#     # place workers
#     print(game._state)
#     game.step(0)
#     game.step(22)
#     game.step(23)
#     game.step(24)
#     print(game._state)
#     #print(game._board.print_state())
#     #print(game._get_current_player().get_valid_actions())
#     game.step((0, 1, 0))
#     game.get_board().print_board()

# if __name__ == "__main__":
#     main()
