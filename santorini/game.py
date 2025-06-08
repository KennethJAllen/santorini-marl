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
        self.board: Board = Board() # The game board, an instance of the Board class
        self.state: GameState = GameState.PLAYER_SELECT
        self.valid_actions: set[int] = {2, 3}
        self._num_workers: int = num_workers
        self.players: list[Player] = [] # List of Player objects participating in the game
        self._current_player_index: int = 0  # Index to keep track of whose turn it is
        self.winner: Player | None = None # the winner of the game

    def reset(self) -> None:
        """Sets the board back to start."""
        self.board = Board()
        self.players = []
        self._current_player_index = 0
        self.state = GameState.PLAYER_SELECT
        self.winner = None # the winner of the game

    def step(self, action: int) -> None:
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

        # Update the valid actions for the next player
        self._update_valid_actions()
        if not self.valid_actions: # if next player has no valid moves
            # TODO: fix this logic for 3 players.
            # If a player has no valid moves, their pieces should be removed from the game.
            # Then, if there is 1 player left, the winner should be declared.
            previous_player_index = (self._current_player_index - 1) % len(self.players)
            self.winner = self.players[previous_player_index]
            self.state = GameState.GAME_OVER

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
        if action not in self.valid_actions:
            raise ValueError(f"Number of players must be one of {', '.join(map(str,self.valid_actions()))}. Instead got: {action}")
        num_players = action
        self._init_players(num_players)
        self.state = GameState.SETUP

    def _handle_setup(self, action: int) -> None:
        """
        Sets player pieces on the board.
        Takes an action which is a position index from 0 to 25.
        """
        if not action in self.valid_actions:
            raise ValueError(f"Invalid action: {utils.decode_action(action)}")

        current_player = self.current_player()
        position = utils.space_index_to_position(action)
        worker_id = len(current_player.get_workers())
        new_worker = Worker(worker_id=worker_id, player=current_player)
        current_player.add_worker(new_worker)
        self.board.place_worker(position, new_worker)

        if len(current_player.get_workers()) >= self._num_workers:
            self._current_player_index += 1
            if self._current_player_index >= len(self.players):
                self._current_player_index = 0
                self.state = GameState.PLAYING

    def _handle_turn(self, action: int) -> None:
        """
        Applies the action in the form of (worker_id, move_index, build_index)
        to the game state if it is a valid action.
        """
        current_player = self.players[self._current_player_index]
        if action not in self.valid_actions:
            raise ValueError(f"Invalid action: {utils.decode_action(action)}")

        move_from, move_to, build_on = utils.decode_action(action)
        did_move_win = self.board.move_worker(move_from, move_to)
        if did_move_win:
            self.state = GameState.GAME_OVER
            self.winner = current_player
        else:
            self.board.build(build_on)
            # cycle through player turns
            self._current_player_index = (self._current_player_index + 1) % len(self.players)

    def _init_players(self, num_players) -> None:
        """Initializes the players in the game."""
        for player_id in range(num_players):
            self.players.append(Player(player_id))

    def _update_valid_actions(self) -> None:
        """
        Returns a set of valid actions for the current player.
        The action space is 5x5x8x8, where:
        - Each of the 5x5 positions identifies the square to move the piece from.
        - The next 64 planes represents a move along one of eight relative compass directions {N, NE, E, SE, S, SW, W, NW}
        along with a build action also along one of eight relative compass directions.
        If the game is in SETUP state,
        - the valid actions are the integer positions on the board where a piece can be placed.
        eIf the game is in PLAYER_SELECT state,
        - the valid actions are the integers 2 or 3, representing the number of players in the game.
        """
        if self.state == GameState.PLAYER_SELECT:
            self.valid_actions =  {2, 3}  # Valid actions are the number of players
        elif self.state == GameState.SETUP:
            self.valid_actions =  self.board.get_valid_placement_actions()
        elif self.state == GameState.PLAYING:
            player = self.current_player()
            valid_actions = set()
            for worker in player.get_workers():
                worker_valid_actions = self.board.get_valid_worker_actions(worker)
                # Add worker's valid actions to total set of player's valid actions
                valid_actions = valid_actions | worker_valid_actions
            self.valid_actions = valid_actions
        else:
            raise ValueError(f"Cannot get valid actions in state: {self.state}")

if __name__ == "__main__":
    g = Game()
    g.step(2)  # Select 2 players
    g.step(0)  # Place first worker at position 0
    g.step(1)  # Place second worker at position 1
    g.step(2)  # Place third worker at position 2
    g.step(3)  # Place fourth worker at position 3
    print(g.valid_actions)
