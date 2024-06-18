"""Defines the Worker and Player classes."""
from __future__ import annotations

from santorini import utils
from santorini.config import SQUARE_SIZE

class Worker:
    """Worker class to represent a player's worker on the board."""

    def __init__(self, worker_id: int = None, player: Player = None, position: tuple[int, int] = None):
        """
        Initializes a new worker with a player, an identifier, and an initial position on the board.
        Default values represent no worker.
        
        player: The player who owns this worker.
        worker_id: An identifier to distinguish between a player's workers (e.g., 1 or 2).
        position: The position that the worker is on.
        gender: Relevant for god cards. Not currently used.
        """
        self._worker_id = worker_id
        self._player = player
        self._position = position
        self._valid_moves = None # The valid locations the worker can move to.
        self._valid_builds = None # The valid locations the worker can build on.

    def __bool__(self):
        return bool(self._player or self._worker_id)

    def __eq__(self, other):
        if not isinstance(other, Worker):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self._player is other._player and self._worker_id == other._worker_id

    def get_player(self) -> Player:
        """Returns the player that the worker belongs to."""
        return self._player

    def set_positon(self, position: tuple[int,int]) -> None:
        """Set the worker position."""
        self._position = position

    def get_position(self) -> tuple[int,int]:
        """Get the worker position."""
        return self._position

    def set_valid_moves(self, valid_moves: set[tuple[int,int]]) -> None:
        """Set the valid moves for the worker."""
        self._valid_moves = valid_moves

    def get_valid_moves(self) -> list[tuple[int,int]]:
        """Get the valid moves for this worker."""
        return self._valid_moves

    def set_valid_builds(self, valid_builds: list[tuple[int,int]]) -> None:
        """Set the valid build locations for the worker."""
        self._valid_builds = valid_builds

    def get_valid_builds(self) -> set[tuple[int,int]]:
        """Get the valid build locations for the worker."""
        return self._valid_builds

    # display

    def draw_piece(self, screen):
        """Display the piece on the screen."""
        if self._player:
            piece_image = self._player.get_piece_image()
            screen.blit(piece_image, self._get_display_position())

    def _get_display_position(self) -> tuple[int,int]:
        """Returns the center of the square corresponding to the worker on the display board."""
        x, y = self._position
        x_display = SQUARE_SIZE * x
        y_display = SQUARE_SIZE * y
        return x_display, y_display

class Player:
    """Player class to manage player actions and workers."""

    def __init__(self, player_id: int = None, workers: dict[int, Worker] = None):
        self._player_id = player_id
        # set of workers
        if workers is None:
            self._workers = []
        else:
            self._workers = workers
        # piece display
        if player_id == 1:
            self._piece_image = utils.load_image('player1.png')
        elif player_id == 2:
            self._piece_image =  utils.load_image('player2.png')
        elif player_id == 3:
            self._piece_image =  utils.load_image('player3.png')
        else:
            self._piece_image = None

    def __bool__(self):
        return bool(self._player_id)

    def get_player_id(self):
        """Reutnrs the id corresponding to the player."""
        return self._player_id

    def add_worker(self, worker: Worker):
        """Adds a worker to the workers dictionary.
        Key (int): worker id, value: worker."""
        self._workers.append(worker)

    def get_workers(self) -> dict[int, Worker]:
        """Get the dictionary of workers.
        Key (int): worker id, value: worker."""
        return self._workers

    def get_piece_image(self) -> tuple[int,int,int]:
        """Returns the image of the player's piece."""
        return self._piece_image
