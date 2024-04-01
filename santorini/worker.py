from santorini.player import Player

class Worker:
    """Worker class to represent a player's worker on the board."""

    def __init__(self, player: Player = None, worker_id: str = "", initial_position: tuple[int, int] = (0, 0)):
        """
        Initializes a new worker with a player, an identifier, and an initial position on the board.
        Default values represent no worker.
        
        :param player: The player who owns this worker.
        :param worker_id: An identifier to distinguish between a player's workers (e.g., 'A' or 'B').
        :param initial_position: A tuple representing the worker's initial position on the board (row, col).
        """
        self.player = player
        self.worker_id = worker_id
        self.position = initial_position
