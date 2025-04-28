"""Main Santorini game state logic"""
import enum
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
    def __init__(self, num_workers: int = NUM_WORKERS):
        self.board = Board() # The game board, an instance of the Board class
        self.state = GameState.PLAYER_SELECT
        self._num_workers = num_workers
        self.players: list[Player] = [] # List of Player objects participating in the game
        self._current_player_index: int = 0  # Index to keep track of whose turn it is
        self.winner: Player = None # the winner of the game

    def reset(self) -> None:
        """Sets the board back to start."""
        self.board = Board()
        self.players = []
        self._current_player_index = 0
        self.state = GameState.PLAYER_SELECT
        self.winner = None # the winner of the game

    def step(self, action) -> None:
        """
        Updates the game with the given action.
        When in the player select phase, the action is an integer representing the number of players in the game.
        When in the setup phase, the action represents a location to place a piece.
        When in the playing phase, 
        """
        if self.state == GameState.PLAYER_SELECT:
            self._handle_player_select(action)
        elif self.state == GameState.SETUP:
            self._handle_setup(action)
        elif self.state == GameState.PLAYING:
            self._handle_turn(action)
        elif self.state == GameState.GAME_OVER:
            # Game over: No actions can be taken
            pass
        else:
            raise ValueError(f"State not handles by 'update_game': {self.state}")

    def is_done(self) -> bool:
        """True if game is over, false otherwise."""
        return self.state == GameState.GAME_OVER

    def current_player(self) -> Player:
        """Returns the current player."""
        if self.players is None:
            raise ValueError("Cannot get the current player. Players are not initialized.")
        return self.players[self._current_player_index]

    def _handle_player_select(self, action: int) -> None:
        """Action is the number of players chosen."""
        valid_actions = [2, 3]
        if action not in valid_actions:
            raise ValueError(f"Number of players must be one of {', '.join(map(str,valid_actions))}. Instead got: {action}")
        num_players = action
        self._init_players(num_players)
        self.state = GameState.SETUP

    def _handle_setup(self, action) -> None:
        """
        Sets player pieces on the board.
        Takes an action which is a position index from 0 to 25.
        """
        current_player = self.current_player()
        valid_actions = self.board.get_valid_placement_actions()
        if not action in valid_actions:
            raise ValueError(f"Invalid action: {action}")

        position = utils.space_index_to_position(action)
        worker_id = len(current_player.get_workers())
        new_worker = Worker(worker_id=worker_id, player=current_player)
        current_player.add_worker(new_worker)
        self.board.place_worker(position, new_worker)

        if len(current_player.get_workers()) >= self._num_workers:
            self._current_player_index += 1
            if self._current_player_index >= len(self.players):
                self._current_player_index = 0
                self._update_current_player_valid_actions()
                self.state = GameState.PLAYING

    def _handle_turn(self, action: tuple[int, int, int]) -> None:
        """
        Applies the action in the form of (worker_id, move_index, build_index)
        to the game state if it is a valid action.
        """
        current_player = self.players[self._current_player_index]
        if tuple(action) not in current_player.get_valid_actions():
            raise ValueError(f"Invalid action: {action}")

        worker_id, move_index, build_index = action

        worker = current_player.get_worker(worker_id)
        move_position = utils.space_index_to_position(move_index)
        did_move_win = self.board.move_worker(worker, move_position)
        if did_move_win:
            self.state = GameState.GAME_OVER
            self.winner = current_player
        else:
            build_position = utils.space_index_to_position(build_index)
            self.board.build(build_position)
            self._cycle_turn()

    def _cycle_turn(self) -> None:
        """Passes the turn to the next player and checks if they have a valid move."""
        previous_player = self.current_player()
        # cycle through player turns
        self._current_player_index = (self._current_player_index + 1) % len(self.players)
        current_player = self.current_player()
        self._update_current_player_valid_actions()
        if not current_player.get_valid_actions(): # if next player has no valid moves
            # TODO: fix this logic for 3 players.
            # If a player has no valid moves, their pieces should be removed from the game.
            # Then, if there is 1 player left, the winner should be declared.
            self.winner = previous_player
            self.state = GameState.GAME_OVER

    def _init_players(self, num_players) -> None:
        """Initializes the players in the game."""
        for player_id in range(num_players):
            self.players.append(Player(player_id))

    def _update_current_player_valid_actions(self) -> None:
        """
        Updates all valid actions the player can take.
        An action is of the form (worker_id, move_index, build_index).
        If a move will win the game (by moving a piece to a height 3 building), the build index is arbitrary.
        """
        player = self.current_player()
        player_valid_actions = set()
        for worker in player.get_workers():
            worker_valid_actions = self.board.get_valid_worker_actions(worker)
            # Add worker's valid actions to total set of player's valid actions
            player_valid_actions = player_valid_actions | worker_valid_actions
        player.set_valid_actions(player_valid_actions)
