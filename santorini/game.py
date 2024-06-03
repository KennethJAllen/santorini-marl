"""Game class containing game logic."""
from santorini.board import Board
from santorini.player import Player, Worker

class Game:
    """Game logic, setup, and main loop."""

    def __init__(self, players: list[Player], board: Board):
        self._board = board  # The game board, an instance of the Board class
        self._players = players # List of Player objects participating in the game
        self._current_player_index = 0  # Index to keep track of whose turn it is
        self._num_workers = 2 # number of workers each player has
        self._num_placed_workers = 0 # number of current player's placed workers. Used in setup
        self._moved_worker = None # tracks the worker moved
        self._game_state = 'setup' # either 'setup', 'playing', or 'game_over' depending on game state.
        self._player_action_sate = 'start_turn' # either 'move', 'build', or 'end_turn' depending on turn player's action state
        self._winner = None

    def get_board(self):
        """Return the game board."""
        return self._board

    def setup_board(self):
        """Prepare the game board for play (e.g., initialize players, place workers)."""
        selected_position = self._board.get_selected_position()
        if selected_position:
            if self._board.can_place(selected_position):
                # If valid space has been selected, add worker to player's workers and set on space.
                player = self._players[self._current_player_index]
                worker_id = self._num_placed_workers
                worker = Worker(player= player, worker_id= worker_id)
                player.add_worker(worker) # add worker to player's list of workers
                self._board.place(selected_position, worker) # sets worker on board
                self._num_placed_workers += 1

                # Handles when maximum number of workers have been placed.
                if self._num_placed_workers == self._num_workers:
                    if self._current_player_index == len(self._players) - 1:
                        # if all workers have been placed
                        self._game_state = 'playing'
                        self._current_player_index = 0
                    else:
                        # if all workers for the current player have been placed
                        self._num_placed_workers = 0
                        self._current_player_index += 1
            self._board.set_selected_position(None)

    def game_loop(self):
        """Main game loop."""
        # initial setup
        if self._game_state == 'setup':
            self.setup_board()

        # main game loop
        elif self._game_state == 'playing':
            if self._player_action_sate == 'start_turn':
                self.start_turn()
            elif self._player_action_sate == 'move':
                self.move_action()
            elif self._player_action_sate == 'build':
                self.build_action()
            elif self._player_action_sate == 'end_turn':
                self.end_turn()
            else:
                raise ValueError("Action state not one of 'start_turn', 'move', 'build', or 'end_turn'.")
        # end game
        elif self._game_state == 'game_over':
            pass
        else:
            raise ValueError("Game state not one of 'setup', playing', or 'game_over'")

    def start_turn(self):
        """
        Start the player's turn by ensuring there is at least one valid move.
        If the current player cannot move, declare the other player the winner.
        This only works for a 2 player implementation.
        """
        player = self._players[self._current_player_index]
        if self._board.check_cannot_move_lose_condition(player):
            winning_player_index = (self._current_player_index + 1) % len(self._players)
            self._winner = self._players[winning_player_index]
            self._game_state = 'game_over'
        else:
            self._player_action_sate = 'move'

    def move_action(self):
        """Moves the selected worker to the selected position."""
        selected_space = self._board.get_selected_position()
        selected_worker = self._board.get_selected_worker()
        turn_player = self._players[self._current_player_index]
        if selected_worker and selected_worker.get_player() is turn_player:
            if selected_space in selected_worker.get_valid_moves():
                worker_position = selected_worker.get_position()
                # execute move
                self._board.move_worker(worker_position, selected_space)
                # track the moved worker
                self._moved_worker = selected_worker
                # check if move won the game
                if self._board.game_over_status():
                    player = self._players[self._current_player_index]
                    self._game_state = 'game_over'
                    self._winner = player
                # update valid build actions and action state
                self.update_valid_build_actions()
                self._player_action_sate = 'build'

    def build_action(self):
        """Moved worker builds on selected space."""
        selected_position = self._board.get_selected_position()
        if selected_position in self._moved_worker.get_valid_builds():
            worker_position = self._moved_worker.get_position()
            # execute build
            self._board.build(worker_position, selected_position)
            # update all valid move actions and action state
            self.update_valid_move_actions()
            self._player_action_sate = 'end_turn'

    def end_turn(self):
        """Passes the turn to the next player."""
        num_players = len(self._players)
        self._current_player_index = (self._current_player_index + 1) % num_players
        self._board.set_selected_worker(None)
        self._board.set_selected_position(None)
        self._player_action_sate = 'start_turn'

    def update_valid_build_actions(self):
        """Updates the moved worker's valid build locations."""
        self._board.update_worker_valid_builds(self._moved_worker) # update valid build location

    def update_valid_move_actions(self):
        """Updates all worker's valid move locations."""
        for player in self._players:
            for worker in player.get_workers():
                self._board.update_worker_valid_moves(worker)

    def get_winner(self):
        """Return the player that won the game."""
        return self._winner
