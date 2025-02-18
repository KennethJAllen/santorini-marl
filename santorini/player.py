"""Defines the Worker and Player classes."""
from __future__ import annotations

class Worker:
    """Worker class to represent a player's worker on the board."""

    def __init__(self, worker_id: int = None, player: Player = None):
        """
        Initializes a new worker with a player, an identifier, and an initial position on the board.
        Default values represent no worker.
        
        player: The player who owns this worker.
        worker_id: An identifier to distinguish between a player's workers (e.g., 1 or 2).
        position: The position that the worker is on.
        gender: Relevant for god cards. Not currently used.
        """
        self._id: int = worker_id
        self._player: Player = player
        self._position: tuple[int, int] = None
        # All valid actions positions which consists of (worker_id, move_index, build_index)
        self._valid_actions: set[tuple[int, int, int]] = None

    def __bool__(self):
        return bool(self._player or self._id)

    def __eq__(self, other):
        if not isinstance(other, Worker):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self._player is other._player and self._id == other._id

    def __repr__(self):
        """Show the player and worker ID
        e.g. "P1W0-H1" => "Player 1's Worker 0"""
        if self._player is None or self._id is None:
            return ""
        return f"P{self._player.get_id()}W{self._id}"

    def get_id(self) -> int:
        """Returns the id of the worker."""
        return self._id

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
        self._valid_actions = valid_moves

    def get_valid_moves(self) -> set[tuple[int,int]]:
        """Get the valid moves for this worker."""
        return self._valid_actions

class Player:
    """Player class to manage player actions and workers."""

    def __init__(self, player_id: int = None, workers: list[Worker] = None):
        self._id = player_id
        self._valid_actions = set()
        # set of workers
        if workers is None:
            self._workers = []
        else:
            self._workers = workers

    def __bool__(self):
        return False if self._id is None else True

    def __repr__(self):
        if self._id is None:
            return ""
        return f"Player {self._id}"

    def get_id(self):
        """Reutnrs the id corresponding to the player."""
        return self._id

    def set_valid_actions(self, valid_actions: set[tuple[int,int,int]]) -> None:
        """Sets the player's valid actions to new valid actions."""
        self._valid_actions = valid_actions

    def get_valid_actions(self) -> set[tuple[int, int, int]]:
        """Returns all the valid actions the player has availible."""
        return self._valid_actions

    def add_worker(self, worker: Worker) -> None:
        """Adds a worker to the list of workers."""
        self._workers.append(worker)

    def get_worker(self, worker_id) -> Worker:
        """Returns the player's worker with corresponding worker_id."""
        for worker in self._workers:
            if worker.get_id() == worker_id:
                return worker
        raise ValueError(f"Player does not have any workers with worker id: {worker_id}")

    def get_workers(self) -> list[Worker]:
        """Get the dictionary of the player's workers."""
        return self._workers
