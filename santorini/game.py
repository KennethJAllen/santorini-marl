"""Game logic and main loop."""
from collections import defaultdict

class Game:
    """Game logic and main loop."""

    def __init__(self, game):
        self.game = game

class Board:
    """Board class to handle the game board and buildings."""
    def __init__(self, board_dims: tuple[int,int] = (5, 5), max_building_height: int = 3, num_players: int = 2):
        """
        Initializes the game board.

        :param dims = [5,5]: The length and width respectively of the board
        :param max_building_height = 3: The maximum height of a non-capped building
        :param num_players = 2: The number of players in the game.
        """
        self.width = int(board_dims[0])
        self.height = int(board_dims[1])
        self.max_building_height = int(max_building_height)
        self.num_players = int(num_players)

        # board state stored as a dict:
        # key: tuple of integers (x,y) representing location on the board
        # value: list of worker and building height
        self.state = defaultdict(lambda: [None, 0])

    def get_state(self):
        """Returns the board state."""
        return self.state

    def display_board(self):
        """Prints the board state to the console."""
        print(self.state)

    def can_move(self, current_position, target_position):
        """
        Checks if a move from current_position to target_position is valid.

        :param current_position: A tuple (xc, yc) indicating the worker's current position.
        :param target_position: A tuple (xt, yt) indicating the desired position to move to.
        :return: True if the move is valid, False otherwise.

        A move is valid if it is:
        1) To an adjacent square on the board
        2) The current position has a worker and the target position does not have a worker
        3) The height of the target_position is at most 1 more than the height of current_position
        """
        xc, yc = current_position
        xt, yt = target_position
        if not(0 <= xc < self.width and 0 <= xt < self.width):
            # check horizontal locations are on board
            return False
        if not(0 <= yc < self.height and 0 <= yt < self.height):
            # check vertical locations are on board
            return False
        if max(abs(xc-xt), abs(yc-yt)) != 1:
            # check target_position is adjacent to current_position
            return False

        current_worker, current_height = self.get_state()[current_position]
        target_worker, target_height = self.get_state()[target_position]

        if not current_worker or target_worker:
            # check there is not a worker on the target_position,
            # and there is a worker on the current_position
            return False

        if target_height > current_height + 1:
            # check target_height is at most 1 more than current_height
            return False

        return True

    def move_worker(self, current_position, target_position):
        """
        Moves a worker from current_position to new_position if the move is valid.

        :param current_position: A tuple (x, y) indicating the worker's current position.
        :param new_position: A tuple (x, y) indicating the new position to move the worker to.
        """
        if not self.can_move(current_position, target_position):
            raise ValueError("That is not a valid move.")

        current_worker = self.get_state()[current_position][0]

        self.state[current_position][0] = None
        self.state[target_position][0] = current_worker

    def can_build(self, worker_position, build_position):
        """
        Checks if building on build_position is valid, given the worker's current position.

        :param worker_position: A tuple (x, y) indicating the worker's current position.
        :param build_position: A tuple (x, y) indicating the position to build on.
        :return: True if the build is valid, False otherwise.
        """

    def build(self, build_position):
        """
        Increments the building level on build_position if the build is valid.

        :param build_position: A tuple (x, y) indicating the position to build on.
        """

    def check_win_condition(self, position):
        """
        Checks if moving to the given position meets the win condition (reaching the third level).

        :param position: A tuple (x, y) indicating the position to check.
        :return: True if the position is on the third level, False otherwise.
        """


class Player:
    """Player class to manage player actions and workers."""

    def __init__(self, player):
        self.player = player

class Worker:
    """Worker class to represent a player's worker on the board."""

    def __init__(self, player, worker_id, initial_position=(0, 0)):
        """
        Initializes a new worker with a player, an identifier, and an initial position on the board.
        
        :param player: The player who owns this worker.
        :param id: An identifier to distinguish between a player's workers (e.g., 'A' or 'B').
        :param initial_position: A tuple representing the worker's initial position on the board (row, col).
        """
        self.player = player
        self.worker_id = worker_id
        self.position = initial_position
