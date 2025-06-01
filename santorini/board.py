"""Defines Board class."""
from collections import defaultdict
from itertools import product
import numpy as np
from player import Worker
import utils
from config import GRID_SIZE, MAX_BUILDING_HEIGHT

class Board:
    """Board class to handle the game board, buildings, board state, and displaying the board."""

    def __init__(self, grid_size = GRID_SIZE, max_building_height = MAX_BUILDING_HEIGHT, state = None):
        """
        Initializes the game board.

        _grid_size: The length and width of the square board. Default 5.
        _max_building_height: The maximum height of a non-capped building. Default 3.
        _select: The space the current player has selected.
        """
        self._grid_size = grid_size
        self._max_building_height = max_building_height

        # board state stored as a dict:
        # key: tuple of integers (x,y) representing location on the board
        # value: list of worker, building height.
        if state is None:
            self._state = defaultdict(lambda: [Worker(), 0])
        else:
            self._state = state

    def __str__(self) -> str:
        """Return a string representing the board with column headers,
        row labels, and column-aligned cells showing worker or height."""
        lines = []

        # Build header with column indices.
        header = "    " + "".join(f"{col:^7}" for col in range(self._grid_size))
        lines.append(header)

        for row in range(self._grid_size):
            # Start each row with the row label.
            row_line = f"{row:<3}"
            for col in range(self._grid_size):
                position = (row, col)
                worker = self.get_position_worker(position)
                height = self.get_position_height(position)

                # Represent capped space with "∞" if needed.
                height_str = "∞" if height > self._max_building_height else str(height)

                if worker:
                    cell_str = f"{worker}-H{height_str}"
                else:
                    cell_str = f"H{height_str}"

                # Append the cell, center-aligned in a field of width 7.
                row_line += f"{cell_str:^7}"
            lines.append(row_line)

        # Append an extra blank line.
        lines.append("")
        return "\n".join(lines)

    def get_grid_size(self) -> int:
        """
        Returns the grid size.
        The board is two dimensional, with each dimension equal to the grid size.
        For example, if the grid size is 5, the dimensions of the board is 5 x 5.
        """
        return self._grid_size

    def move_worker(self, worker: Worker, target_position: tuple[int, int]) -> bool:
        """
        Moves a worker to new_position.
        worker: A Worker to move.
        new_position: A tuple (x, y) indicating the new position to move the worker to.
        Returns True if the move was to a winning height, False otherwise.
        """
        worker_position = worker.get_position()
        self._set_position_worker(worker_position, Worker()) # default worker represents no worker
        self._set_position_worker(target_position, worker)
        did_move_win = self.get_position_height(target_position) == self._max_building_height
        return did_move_win

    def build(self, build_position: tuple[int, int])  -> None:
        """
        Increment the height of build_position.
        Assumes the check that the build is valid happens when the move is validated.
        """
        self._state[build_position][1] += 1

    def place_worker(self, position: tuple[int, int], worker: Worker) -> None:
        """Place the worker on the location"""
        self._set_position_worker(position, worker)

    def get_valid_placement_actions(self) -> set[int]:
        """Gets all valid_locations where a worker can be placed"""
        valid_actions = set()
        for i, j in product(range(self._grid_size), range(self._grid_size)):
            if not self.get_position_worker((i, j)):
                action = utils.space_position_to_index((i, j))
                valid_actions.add(action)
        return valid_actions

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

    def get_observation(self, current_player_index) -> np.ndarray:
        """
        Returns an array-based representation of the board state
        shape=(5,5,3)
        channel 1: building height
        channel 2: which player's piece occupies each cell. 0 for player 1, 1 for player2, -1 if empty.
        channel 3: Who is the turn player? All zeros for first player's turn, all ones for second player's turn.
        """
        obs = np.empty((self._grid_size, self._grid_size, 3))
        for i, j in product(range(self._grid_size), range(self._grid_size)):
            # channel 1
            obs[i, j, 0] = self.get_position_height((i, j))

            # channel 2
            player = self.get_position_worker((i, j)).get_player()
            if player is None:
                obs[i, j, 1] = 0
            else:
                obs[i, j, 1] = player.get_id() + 1
        # channel 3
        obs[:, :, 2] = current_player_index
        return obs

    def get_position_worker(self, position: tuple[int, int]) -> Worker:
        """Returns the worker in the given position."""
        return self._state[position][0]

    def get_position_height(self, position: tuple[int, int]) -> int:
        """Returns the building height at the given position.
        If the position is capped, returns max height + 1."""
        return self._state[position][1]

    def _get_valid_moves_from_position(self, position: tuple[int, int]) -> list[tuple[int, int]]:
        """Returns set of valid moves that a worker in position can move to."""
        x, y = position
        # check all adjacent positions for valid moves
        valid_moves = []
        for i, j in product(range(-1,2), range(-1,2)):
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
        It is important to track which worker moved to the given position,
        because if a worker moves they can build on their previously occupied space.
        """
        valid_positions = []
        # If the move would result in a win, can "build" anywhere
        if self.get_position_height(position) == self._max_building_height:
            for row in range(self._grid_size):
                for col in range(self._grid_size):
                    target_position = (row, col)
                    valid_positions.append(target_position)
            return valid_positions

        x, y = position
        # check all adjacent positions for valid builds
        for i, j in product(range(-1,2), range(-1,2)):
            target_position = x + i, y + j
            # Must build on board
            if not self._is_on_board(position):
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

    def _set_position_worker(self, position: tuple[int, int], worker: Worker) -> None:
        """Sets the given worker on the given position."""
        self._state[position][0] = worker
        worker.set_positon(position)

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

    def _is_on_board(self, position: tuple[int, int]) -> bool:
        """Returns true if position is on the board. Returns false otherwise."""
        x, y = position
        if 0 <= x < self._grid_size and 0 <= y < self._grid_size:
            return True
        return False
