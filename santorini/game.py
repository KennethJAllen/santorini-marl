"""Game logic and main loop."""
from collections import defaultdict
from santorini.worker import Worker

class Game:
    """Game logic and main loop."""

    def __init__(self, game):
        self.game = game

class Board:
    """Board class to handle the game board and buildings."""

    def __init__(self, grid_size: int = 5, max_building_height: int = 3, num_players: int = 2):
        """
        Initializes the game board.

        :param grid_size = 5: The length and width of the square board
        :param max_building_height = 3: The maximum height of a non-capped building
        :param num_players = 2: The number of players in the game.
        """
        self._grid_size = int(grid_size)
        self._max_building_height = int(max_building_height)
        self._num_players = int(num_players)

        # board state stored as a dict:
        # key: tuple of integers (x,y) representing location on the board
        # value: list of worker and building height.
        self._state = defaultdict(lambda: [Worker(), 0])

    def get_grid_size(self) -> int:
        """
        Returns the grid size.
        The board is two dimensional, with each dimension equal to the grid size.
        For example, if the grid size is 5, the dimensions of the board is 5 x 5.
        """
        return self._grid_size

    def get_position_worker(self, position: tuple[int, int]) -> Worker:
        """Returns the worker in the given position."""
        return self._state[position][0]

    def set_position_worker(self, position: tuple[int, int], worker: Worker) -> None:
        """Sets the given worker on the given position."""
        self._state[position][0] = worker

    def get_position_height(self, position: tuple[int, int]) -> int:
        """Returns the building height at the given position.
        If the position is capped, returns math.inf."""
        return self._state[position][1]

    def set_position_height(self, position: tuple[int, int], height: int) -> None:
        """Sets the given height on the given position."""
        self._state[position][1] = height

    def can_move(self, current_position: tuple[int, int], target_position: tuple[int,int]) -> bool:
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
        grid_size = self.get_grid_size()
        if not(0 <= xc < grid_size and 0 <= xt < grid_size):
            # check horizontal locations are on board
            return False

        if not(0 <= yc < grid_size and 0 <= yt < grid_size):
            # check vertical locations are on board
            return False

        if max(abs(xc - xt), abs(yc - yt)) != 1:
            # check target_position is adjacent to current_position
            return False

        current_worker = self.get_position_worker(current_position)
        target_worker = self.get_position_worker(target_position)
        if not current_worker or target_worker:
            # check there is not a worker on the target_position,
            # and there is a worker on the current_position
            return False

        current_height = self.get_position_height(current_position)
        target_height = self.get_position_height(target_position)
        if target_height > current_height + 1:
            # check target_height is at most 1 more than current_height
            return False

        return True

    def move_worker(self, current_position: tuple[int, int], target_position: tuple[int, int]) -> None:
        """
        Moves a worker from current_position to new_position if the move is valid.

        :param current_position: A tuple (x, y) indicating the worker's current position.
        :param new_position: A tuple (x, y) indicating the new position to move the worker to.
        """
        if not self.can_move(current_position, target_position):
            raise ValueError("That is not a valid move.")

        worker = self.get_position_worker(current_position)
        self.set_position_worker(current_position, Worker()) # Worker with no parameters represents no worker.
        self.set_position_worker(target_position, worker)

    def can_build(self, worker_position: tuple[int, int], build_position: tuple[int, int]) -> bool:
        """
        Checks if building on build_position is valid, given the worker's current position.

        :param worker_position: A tuple (x, y) indicating the worker's current position.
        :param build_position: A tuple (x, y) indicating the position to build on.
        :return: True if the build is valid, False otherwise.
        """

    def build(self, build_position: tuple[int, int]) -> None:
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

    def display(self):
        """Prints the board state to the console."""
        grid_size = self.get_grid_size()
        board = []
        for row_index in range(grid_size):
            row = []
            for col_index in range(grid_size):
                position = (row_index, col_index)
                worker = self.get_position_worker(position)
                height = self.get_position_height(position)
                row.append((str(worker.player), str(height)))
            board.append(row)
        print(board)

board = Board()
worker_a = Worker(player='A')
worker_b = Worker(player='B')
board.set_position_worker((1,1), worker_a)
board.set_position_worker((1,2), worker_a)
board.set_position_worker((2,2), worker_b)
board.set_position_worker((3,3), worker_b)
board.set_position_height((2,2), 2)
board.set_position_height((4,4), 1)
board.display()
