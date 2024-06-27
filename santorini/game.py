"""Defines Game class."""
import pygame
from santorini.board import Board
from santorini.player import Player, Worker
from santorini import utils
from santorini.config import NUM_WORKERS, WIDTH, HEIGHT, BLACK

class Game:
    """Game logic, setup, and main loop."""

    def __init__(self, screen):
        self._screen = screen # The display screen
        self._board = Board()  # The game board, an instance of the Board class
        self._players = None # List of Player objects participating in the game
        self._current_player_index = 0  # Index to keep track of whose turn it is
        self._num_placed_workers = 0 # number of current player's placed workers. Used in setup
        self._moved_worker = None # tracks the worker moved
        self._game_state = 'player_select' # The game state.
        self._player_action_sate = 'start_turn' # either 'start_turn', 'move', 'build', or 'end_turn' depending on turn player's action state
        self._winner = None # the winner of the game

    def game_loop(self, display_position: tuple[int,int]):
        """Main game loop."""
        # choose number of players
        if self._game_state == 'player_select':
            self._player_select(display_position)
            
        # initial setup
        elif self._game_state == 'setup':
            self._select(display_position)
            self._setup_board()

        # main game loop
        elif self._game_state == 'playing':
            self._select(display_position)
            if self._player_action_sate == 'start_turn':
                self._start_turn()
            if self._player_action_sate == 'move':
                self._move_action()
            elif self._player_action_sate == 'build':
                self._build_action()
            if self._player_action_sate == 'end_turn':
                self._end_turn()
        # end game
        elif self._game_state == 'game_over':
            self._setup()

    def _setup(self):
        """Sets the board back to blank."""
        self._board = Board()
        self._current_player_index = 0  # Index to keep track of whose turn it is
        self._num_placed_workers = 0 # number of current player's placed workers. Used in setup
        self._moved_worker = None # tracks the worker moved
        self._game_state = 'setup' # The game state.
        self._player_action_sate = 'start_turn' # either 'start_turn', 'move', 'build', or 'end_turn' depending on turn player's action state
        self._winner = None # the winner of the game

    def display_game(self):
        """Displays the current board state."""
        if self._game_state == 'game_over':
            self._display_game_over_screen()
        elif self._game_state == 'player_select':
            self._display_start_screen()
        else:
            self._display_board()
        pygame.display.update()

    # game state methods

    def _select(self, display_position):
        """Update the selected location and worker."""
        position = utils.convert_to_position(display_position)
        self._board.set_selected_position(position)
        self._board.set_selected_worker(position)

    def _player_select(self, display_position: tuple[int,int]):
        """Choose the number of players in the game."""
        x_position = display_position[0]
        if x_position <= WIDTH//3:
            num_players = 1
        elif x_position <= 2*WIDTH//3:
            num_players = 2
        else:
            num_players = 3
        self._init_players(num_players)
        self._game_state = 'setup'

    def _init_players(self, num_players):
        """Initializes the players in the game."""
        self._players = []
        for player_id in range(1,num_players+1):
            self._players.append(Player(player_id))
        # if num_players == 1:
        #     self._players.append(Player(2, ai = True))

    def _setup_board(self):
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
                if self._num_placed_workers == NUM_WORKERS:
                    if self._current_player_index == len(self._players) - 1:
                        # if all workers have been placed
                        self._update_all_valid_move_actions()
                        self._game_state = 'playing'
                        self._current_player_index = 0
                    else:
                        # if all workers for the current player have been placed
                        self._num_placed_workers = 0
                        self._current_player_index += 1
            self._board.set_selected_position(None)

    def _start_turn(self):
        """Starts turn."""
        self._player_action_sate = 'move'

    def _move_action(self):
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
                self._board.update_worker_valid_builds(self._moved_worker)
                self._player_action_sate = 'build'

    def _build_action(self):
        """Moved worker builds on selected space."""
        selected_position = self._board.get_selected_position()
        if selected_position in self._moved_worker.get_valid_builds():
            worker_position = self._moved_worker.get_position()
            # execute build
            self._board.build(worker_position, selected_position)
            # update all valid move actions and action state
            self._update_all_valid_move_actions()
            self._player_action_sate = 'end_turn'

    def _end_turn(self):
        """Passes the turn to the next player."""
        num_players = len(self._players)
        current_player = self._players[self._current_player_index]
        self._current_player_index = (self._current_player_index + 1) % num_players
        #self._board.set_selected_worker(None)
        #self._board.set_selected_position(None)
        self._player_action_sate = 'start_turn'
        # check that next player has a valid move
        next_player = self._players[self._current_player_index]
        self._board.reset_positions()
        if self._board.check_cannot_move_lose_condition(next_player):
            self._winner = current_player
            self._game_state = 'game_over'

    def _update_all_valid_move_actions(self):
        """Updates all worker's valid move locations."""
        for player in self._players:
            for worker in player.get_workers():
                self._board.update_worker_valid_moves(worker)

    # display

    def _highlight_moves(self):
        """Displays the selected worker and potential moves."""
        worker = self._board.get_selected_worker()
        turn_player = self._players[self._current_player_index]
        if worker and worker.get_player() == turn_player:
            # highlight worker
            worker_position = worker.get_position()
            self._board.display_worker_highlight(worker_position, self._screen)
            # re-desplay worker on top of highlight
            self._board.display_worker(worker_position, self._screen)
            # highlight moves
            for move_location in worker.get_valid_moves():
                self._board.display_move_hightlight(move_location, self._screen)


    def _highlight_builds(self):
        """Displays potential spaces the moved worker can build on."""
        worker = self._moved_worker
        for build_location in worker.get_valid_builds():
            self._board.display_build_hightlight(build_location, self._screen)

    def _display_board(self):
        """Displays the game board"""
        grid_size = self._board.get_grid_size()
        for row_index in range(grid_size):
            for col_index in range(grid_size):
                position = (row_index, col_index)
                self._board.display_building(position, self._screen)
                self._board.display_worker(position, self._screen)
        if self._player_action_sate == 'move':
            self._highlight_moves()
        if self._player_action_sate == 'build':
            self._highlight_builds()

    def _display_game_over_screen(self):
        """Displays the game over screen."""
        # black background
        self._screen.fill(BLACK)

        # game over text
        game_over_text = f"Player {self._winner.get_player_id()} wins!"
        game_over_position = (WIDTH//2, HEIGHT//3)
        utils.display_text(game_over_text, game_over_position, self._screen)

        # restart text
        restart_position = (WIDTH//2, 2*HEIGHT//3)
        utils.display_text('Press to restart', restart_position, self._screen)

    def _display_start_screen(self):
        """Displays the game over screen."""
        # black background
        self._screen.fill(BLACK)

        # game over text
        choose_players_position = (WIDTH//2, HEIGHT//4)
        utils.display_text('Choose # Players', choose_players_position, self._screen)

        # num players text
        one_position = (WIDTH//6, HEIGHT//2)
        two_position = (WIDTH//2, HEIGHT//2)
        three_position = (5*WIDTH//6, HEIGHT//2)
        utils.display_text('1', one_position, self._screen)
        utils.display_text('2', two_position, self._screen)
        utils.display_text('3', three_position, self._screen)
