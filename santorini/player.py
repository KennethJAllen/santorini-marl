"""Defines the Worker and Player classes."""
from __future__ import annotations

class Worker:
    """Worker class to represent a player's worker on the board."""

    def __init__(self, worker_id: int = None, player: Player = None, gender = ""):
        """
        Initializes a new worker with a player, an identifier, and an initial position on the board.
        Default values represent no worker.
        
        :param player: The player who owns this worker.
        :param worker_id: An identifier to distinguish between a player's workers (e.g., 'A' or 'B').
        :param position: The position that the worker is on.
        :param gender: Relevant for god cards. Not currently used.
        """
        self._worker_id = worker_id
        self._player = player
        self._gender = gender

    def __bool__(self):
        return bool(self._player or self._worker_id)

    def __eq__(self, other):
        if not isinstance(other, Worker):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self._player is other._player and self._worker_id == other._worker_id and self._gender == other._gender

    def display(self):
        """Displayers the worker info."""
        if self:
            return f"{self._player.player_id}_{self._worker_id}"
        return ""

    def get_worker_id(self):
        """Returns the worker id."""
        return self._worker_id
    
    def get_player(self):
        """Returns the player that the worker belongs to."""
        return self._player

class Player:
    """Player class to manage player actions and workers."""

    def __init__(self, player_id: int = None, workers: dict[int, Worker] = None, god_card = None):
        self.player_id = player_id
        self._god_card = god_card
        if workers is None:
            self._workers = {}
        else:
            self._workers = workers

    def __bool__(self):
        return bool(self.player_id)

    def add_worker(self, worker: Worker):
        """Adds a worker to the workers dictionary. Key: worker id, value: worker."""
        worker_id = worker.get_worker_id()
        self._workers[worker_id] = worker

    def get_worker(self, worker_id):
        """Returns the player's worker with the assosiated worker id."""
        return self._workers[worker_id]
