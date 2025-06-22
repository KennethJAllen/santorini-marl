"""Defines Board class."""
from collections import defaultdict
from itertools import product
import numpy as np
from santorini.player import Worker
from santorini import utils
from santorini.config import GRID_SIZE, MAX_BUILDING_HEIGHT

class Board:
    """Board class to handle the game board, buildings, board state, and displaying the board."""

    def __init__(self, grid_size = GRID_SIZE, max_building_height = MAX_BUILDING_HEIGHT, state = None):
        """
        Initializes the game board.

        _grid_size: The length and width of the square board. Default 5.
        _max_building_height: The maximum height of a non-capped building. Default 3.
        _select: The space the current player has selected.
        """
        self.grid_size = grid_size
        self.max_building_height = max_building_height

        # board state stored as a dict:
        # key: tuple of integers (x,y) representing location on the board
        # value: list of worker, building height.
        if state is None:
            self._state = defaultdict(lambda: [Worker(), 0])
        else:
            self._state = state

    def __str__(self) -> str:
        """
        Return a string representing the board with column headers,
        x labels, and column-aligned cells showing worker or height.
        """
        lines = []

        # Build header with column indices.
        header = "    " + "".join(f"{col:^7}" for col in range(self.grid_size))
        lines.append(header)

        for y in range(self.grid_size):
            row_line = f"{y:<3}"
            for x in range(self.grid_size):
                position = (x, y)
                worker = self.get_position_worker(position)
                height = self.get_height(position)

                # Represent capped space with "∞"
                height_str = "∞" if height > self.max_building_height else str(height)

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

    def move_worker(self, worker_position: tuple[int, int], target_position: tuple[int, int]) -> bool:
        """
        Moves a worker to new_position.
        worker: A Worker to move.
        new_position: A tuple (x, y) indicating the new position to move the worker to.
        Returns True if the move was to a winning height, False otherwise.
        """
        worker = self.get_position_worker(worker_position)
        self._set_position_worker(worker_position, Worker()) # Worker with no args represents no worker.
        self._set_position_worker(target_position, worker)
        did_move_win = self.get_height(target_position) == self.max_building_height
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
        for i, j in product(range(self.grid_size), range(self.grid_size)):
            if not self.get_position_worker((i, j)):
                action = utils.encode_space((i, j))
                valid_actions.add(action)
        return valid_actions

    def get_valid_worker_actions(self, worker: Worker) -> set[int]:
        """
        Updates all valid actions the player can take.
        An action is an integer from 0 to 5*5*8*8.
        If a move will win the game (by moving a piece to a height 3 building), the build index is arbitrary.
        """
        valid_actions = set()
        worker_position = worker.position
        valid_moves = self._get_valid_moves_from_position(worker_position)
        for move_position in valid_moves:
            for build_position in self._valid_move_then_build_positions(worker, move_position):
                action = utils.encode_action((worker_position, move_position, build_position))
                valid_actions.add(action)
        return valid_actions

    def get_observation(self, current_player_index) -> np.ndarray:
        """
        Returns an array-based representation of the board state.
        The board is represented by 7 binary channels,
        each of size grid_size x grid_size (typically 5x5).

        channel 0-4: building heights.
            - 0: 1 nothing if no building, else 0
            - 1: 1 for height 1, else 0
            - 2: 1 for height 2, else 0
            - 3: 1 for height 3, else 0
            - 4: 1 if capped, else 0
            (max building height greater than 3 is not supported by this function)
        channel 5-6: which player's piece occupies each cell.
            - 5: 1 if the turn player occupies the cell, else 0
            - 6: 1 if the other player occupies the cell, else 0

        WARNING: This function assumes that the game is played with 2 players.
        """
        obs = np.zeros((self.grid_size, self.grid_size, 7), dtype=bool)
        # iterate through each position on the board
        for i, j in product(range(self.grid_size), range(self.grid_size)):
            height = self.get_height((i, j))
            # channels 0-4:
            # Just set the channel corresponding to the height to 1
            # because default is 0 and channel i corresponds to height i.
            obs[i, j, height] = 1
            # channel 5-6:
            player = self.get_position_worker((i, j)).get_player()
            if player is None:
                continue
            if player.get_id() == current_player_index:
                # channel 5: current player
                obs[i, j, 5] = 1
            else:
                # channel 6: other player
                obs[i, j, 6] = 1
        return obs

    def get_position_worker(self, position: tuple[int, int]) -> Worker:
        """Returns the worker in the given position."""
        return self._state[position][0]

    def get_height(self, position: tuple[int, int]) -> int:
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
        build_positions = []
        x, y = position
        # check all adjacent positions for valid builds
        for i, j in product(range(-1,2), range(-1,2)):
            build_position = x + i, y + j
            # Must build on board
            if not self._is_on_board(build_position):
                continue
            # Can't build on current position
            # Equivalent to when i = j = 0
            if build_position == position:
                continue
            # If the move would result in a win, can "build" on any adjacent position
            if self.get_height(position) == self.max_building_height:
                build_positions.append(build_position)
                continue
            # Check if target position is capped
            position_height = self.get_height(build_position)
            if position_height == self.max_building_height + 1:
                continue
            # Check if there is a different worker on the target position
            target_position_worker = self.get_position_worker(build_position)
            if target_position_worker and not (target_position_worker is worker):
                continue

            build_positions.append(build_position)
        return build_positions

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

        current_height = self.get_height(worker_position)
        target_height = self.get_height(target_position)
        if target_height > current_height + 1:
            # check target_height is at most 1 more than current_height
            return False
        return True

    def _is_on_board(self, position: tuple[int, int]) -> bool:
        """Returns true if position is on the board. Returns false otherwise."""
        x, y = position
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            return True
        return False
