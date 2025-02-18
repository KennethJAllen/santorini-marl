"""Defines Board class."""
from collections import defaultdict
from santorini.player import Worker
from santorini import utils
from santorini.config import GRID_SIZE, MAX_BUILDING_HEIGHT

class Board:
    """Board class to handle the game board, buildings, board state, and displaying the board."""

    def __init__(self):
        """
        Initializes the game board.

        _grid_size: The length and width of the square board. Default 5.
        _max_building_height: The maximum height of a non-capped building. Default 3.
        _game_over: Flag if the game is over. Default False.
        _select: The space the current player has selected.
        """
        self._grid_size = GRID_SIZE
        self._max_building_height = MAX_BUILDING_HEIGHT
        self._game_over = False

        # board state stored as a dict:
        # key: tuple of integers (x,y) representing location on the board
        # value: list of worker, building height.
        self._state = defaultdict(lambda: [Worker(), 0])

    def get_grid_size(self) -> int:
        """
        Returns the grid size.
        The board is two dimensional, with each dimension equal to the grid size.
        For example, if the grid size is 5, the dimensions of the board is 5 x 5.
        """
        return self._grid_size

    def move_worker(self, worker: Worker, target_position: tuple[int, int]) -> None:
        """
        Moves a worker to new_position.
        worker: A Worker to move.
        new_position: A tuple (x, y) indicating the new position to move the worker to.
        """
        worker_position = worker.get_position()
        self._set_position_worker(worker_position, Worker()) # default worker represents no worker
        self._set_position_worker(target_position, worker)
        self._check_height_win_condition(target_position)

    def build(self, build_position: tuple[int, int])  -> None:
        """
        Increment the height of build_position.
        Assumes the check that the build is valid happens when the move is validated.
        """
        self._state[build_position][1] += 1

    def can_place(self, position: tuple[int, int]) -> bool:
        """Check if placing worker on position is valid."""
        # Check positions are on the board
        if not self._is_on_board(position):
            return False
        # Check worker is on position
        if self.get_position_worker(position):
            return False
        return True

    def place_worker(self, position: tuple[int, int], worker: Worker) -> None:
        """Place the worker on the location."""
        self._set_position_worker(position, worker)

    def get_valid_worker_actions(self, worker: Worker) -> set[tuple[int, int, int]]:
        """
        Updates all valid actions the player can take.
        An action is of the form (worker_id, move_index, build_index).
        If a move will win the game (by moving a piece to a height 3 building), the build index is arbitrary.
        """
        valid_actions = set()
        worker_position = worker.get_position()
        valid_moves = self._get_valid_moves_from_position(worker_position)
        for move_position in valid_moves:
            for build in self._valid_move_then_build_positions(worker, move_position):
                move_index = utils.space_position_to_index(move_position)
                build_index = utils.space_position_to_index(build)
                valid_actions.add((worker.get_id(), move_index, build_index))
        return valid_actions

    def _get_valid_moves_from_position(self, position: tuple[int, int]) -> list[tuple[int, int]]:
        """Returns set of valid moves that a worker in worker position can move to."""
        x, y = position
        # check all adjacent positions for valid moves
        valid_moves = []
        for i in range(-1,2):
            for j in range(-1,2):
                target_position = x + i, y + j
                if target_position == position:
                    continue
                if self._can_move(position, target_position):
                    valid_moves.append(target_position)
        return valid_moves

    def _valid_move_then_build_positions(self, worker: Worker, position: tuple[int, int]) -> list[tuple[int,int]]:
        """
        If the worker moves to the given position,
        returns a list of valid positions for the worker to build.
        """
        x, y = position
        # check all adjacent positions for valid builds
        valid_positions = []
        for i in range(-1,2):
            for j in range(-1,2):
                target_position = x + i, y + j
                # Must build on board
                if not self._is_on_board(position):
                    continue
                # If the move would result in a win, can "build" anywhere
                if self.get_position_height(position) == MAX_BUILDING_HEIGHT:
                    valid_positions.append(target_position)
                    continue
                # Can't build on current position
                if target_position == position:
                    continue
                # Check if target position is capped
                position_height = self.get_position_height(target_position)
                if position_height == self._max_building_height + 1:
                    continue
                # Check if there is a different worker on the target position
                target_position_worker = self.get_position_worker(target_position)
                if target_position_worker and not (target_position_worker is worker):
                    continue
                valid_positions.append(target_position)
        return valid_positions

    def is_game_over(self):
        """Returns True if a player has won, False otherwise."""
        return self._game_over

    def get_position_worker(self, position: tuple[int, int]) -> Worker:
        """Returns the worker in the given position."""
        return self._state[position][0]

    def _set_position_worker(self, position: tuple[int, int], worker: Worker) -> None:
        """Sets the given worker on the given position."""
        self._state[position][0] = worker
        worker.set_positon(position)

    def get_position_height(self, position: tuple[int, int]) -> int:
        """Returns the building height at the given position.
        If the position is capped, returns max height + 1."""
        return self._state[position][1]

    def _set_position_height(self, position: tuple[int, int], height: int) -> None:
        """Sets the given height on the given position."""
        self._state[position][1] = height

    def _can_move(self, worker_position: tuple[int, int], target_position: tuple[int,int]) -> bool:
        """
        Checks if a move from worker_position to target_position is valid.

        worker_position: A tuple (xc, yc) indicating the worker's current position.
        target_position: A tuple (xt, yt) indicating the desired position to move to.
        return: True if the move is valid, False otherwise.

        A move is valid if it is:
        1) To an adjacent square on the board
        2) The current position has a worker and the target position does not have a worker
        3) The height of the target_position is at most 1 more than the height of worker_position
        """
        if not self._is_on_board(worker_position) or not self._is_on_board(target_position):
            # ensure that positions are on board
            return False

        if not utils.is_adjacent(worker_position, target_position):
            return False

        current_worker = self.get_position_worker(worker_position)
        target_worker = self.get_position_worker(target_position)
        if not current_worker or target_worker:
            # check there is not a worker on the target_position,
            # and there is a worker on the worker_position
            return False

        current_height = self.get_position_height(worker_position)
        target_height = self.get_position_height(target_position)
        if target_height > current_height + 1:
            # check target_height is at most 1 more than current_height
            return False
        return True

    def _check_height_win_condition(self, position: tuple[int,int]):
        """
        Checks if moving to the given position meets the win condition (reaching the third level).

        target_position: A tuple (x, y) indicating the position to check.
        return: True if the position is on the third level, False otherwise.
        """
        if self.get_position_height(position) == self._max_building_height:
            self._game_over = True

    def _is_on_board(self, position: tuple[int, int]) -> bool:
        """Returns true if position is on the board. Returns false otherwise."""
        x, y = position
        if 0 <= x < self._grid_size and 0 <= y < self._grid_size:
            return True
        return False

    def _set_state(self, state_data: dict[tuple[int, int], list[Worker, int]]) -> None:
        """
        Sets the board state corresponding to the given state data.
        This is used for testing.
        """
        self._state = state_data

def print_board(board: Board) -> None:
    """Prints the board with column headers, row labels,
    and column-aligned cells showing worker or height."""

    # Print top column indices
    print("    ", end="")  # Leading spaces for alignment
    for col in range(GRID_SIZE):
        print(f"{col:^7}", end="")  # Center each column header in 7 chars
    print()

    for row in range(GRID_SIZE):
        # Print row index on the left
        print(f"{row:<3}", end="")  # Left-align row index in 3 chars

        for col in range(GRID_SIZE):
            position = (row, col)
            worker = board.get_position_worker(position)
            height = board.get_position_height(position)

            # Convert height to a string. Represent capped space with "∞".
            height_str = "∞" if height > MAX_BUILDING_HEIGHT else str(height)

            if worker:
                cell_str = f"{worker}-H{height_str}"
            else:
                cell_str = f"H{height_str}"

            # Center-align each cell in 7 chars
            print(f"{cell_str:^7}", end="")

        print()  # Newline after each row
    print()  # Extra blank line after the board
