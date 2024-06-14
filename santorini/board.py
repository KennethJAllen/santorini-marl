from collections import defaultdict
import math

from santorini.player import Worker, Player
from santorini import utils

class Board:
    """Board class to handle the game board, buildings, board state, and displaying the board."""

    def __init__(self, grid_size: int = 5, max_building_height: int = 3):
        """
        Initializes the game board.

        _grid_size: The length and width of the square board. Default 5.
        _max_building_height: The maximum height of a non-capped building. Default 3.
        _game_over: Flag if the game is over. Default False.
        _select: The space the current player has selected.
        """
        self._grid_size = int(grid_size)
        self._max_building_height = int(max_building_height)
        self._game_over = False
        self._selected_position = None
        self._selected_worker = None

        # board state stored as a dict:
        # key: tuple of integers (x,y) representing location on the board
        # value: list of worker and building height.
        self._state = defaultdict(lambda: [Worker(), 0])

        # display
        self._images = {0: utils.load_image('level0.png'),
                        1: utils.load_image('level1.png'),
                        2: utils.load_image('level2.png'),
                        3: utils.load_image('level3.png'),
                        math.inf: utils.load_image('dome.png'),
                        'selected': utils.load_image('highlight_selected.png'),
                        'move': utils.load_image('highlight_moves.png'),
                        'build': utils.load_image('highlight_build.png')}

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
        if worker:
            worker.set_positon(position)
            self.update_worker_valid_moves(worker)

    def get_position_height(self, position: tuple[int, int]) -> int:
        """Returns the building height at the given position.
        If the position is capped, returns math.inf."""
        return self._state[position][1]

    def set_position_height(self, position: tuple[int, int], height: int) -> None:
        """Sets the given height on the given position."""
        self._state[position][1] = height

    def set_state(self, state_data: dict[tuple[int, int], list[Worker, int]]) -> None:
        """
        Sets the board state corresponding to the given state data.
        This is primarily used for testing.
        """
        self._state = state_data

    def is_on_board(self, position: tuple[int, int]) -> bool:
        """Returns true if position is on the board. Returns false otherwise."""
        grid_size = self.get_grid_size()
        x, y = position
        if 0 <= x < grid_size and 0 <= y < grid_size:
            return True
        return False

    def can_move(self, worker_position: tuple[int, int], target_position: tuple[int,int]) -> bool:
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
        if not self.is_on_board(worker_position) or not self.is_on_board(target_position):
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

    def move_worker(self, current_position: tuple[int, int], target_position: tuple[int, int]) -> None:
        """
        Moves a worker from current_position to new_position.

        current_position: A tuple (x, y) indicating the worker's current position.
        new_position: A tuple (x, y) indicating the new position to move the worker to.
        return: True if the move won the game, false otherwise.
        """
        worker = self.get_position_worker(current_position)
        self.set_position_worker(current_position, Worker()) # default param worker represents no worker
        self.set_position_worker(target_position, worker)
        self.check_height_win_condition(target_position)

    def can_build(self, worker_position: tuple[int, int], build_position: tuple[int, int]) -> bool:
        """
        Checks if building on build position is valid, given the worker's current position.

        worker_position: A tuple indicating the worker's current position.
        build_position: A tuple indicating the position to build on.
        return: True if the build is valid, False otherwise.
        """
        # Check positions are on the board
        if not self.is_on_board(worker_position) or not self.is_on_board(build_position):
            return False

        # Check build position is adjacent to worker position
        if not utils.is_adjacent(worker_position, build_position):
            return False

        # Check if building height has reached a dome
        position_height = self.get_position_height(build_position)
        if position_height == math.inf:
            return False

        # Check if worker is on position
        if self.get_position_worker(build_position):
            return False

        return True

    def build(self, worker_position: tuple[int, int], build_position: tuple[int, int]) -> None:
        """
        Increments the building level on build_position if the build is valid.

        build_position: A tuple (x, y) indicating the position to build on.
        """
        if not self.can_build(worker_position, build_position):
            raise ValueError("That is not a valid build position.")

        position_height = self.get_position_height(build_position)
        if position_height < self._max_building_height:
            self.set_position_height(build_position, position_height + 1)
        # cap if height is max
        else:
            self.set_position_height(build_position, math.inf)

    def can_place(self, position: tuple[int, int]) -> bool:
        """Check if placing worker on position is valid."""
        # Check positions are on the board
        if not self.is_on_board(position):
            return False
        # Check worker is on position
        if self.get_position_worker(position):
            return False
        return True

    def place(self, position: tuple[int, int], worker: Worker) -> None:
        """Place the worker on the location."""
        self.set_position_worker(position, worker) # sets worker on board

    def update_worker_valid_moves(self, worker: Worker) -> None:
        """Returns set of valid moves that a worker in worker position can move to."""
        if worker:
            x, y = worker.get_position()
            # check all adjacent positions for valid moves
            valid_moves = set()
            for i in range(-1,2):
                for j in range(-1,2):
                    target_position = x + i, y + j
                    if (i != 0 or j != 0) and self.can_move((x, y), target_position):
                        valid_moves.add(target_position)
            worker.set_valid_moves(valid_moves)

    def update_worker_valid_builds(self, worker: Worker) -> None:
        """Returns set of valid moves that a worker in worker position can move to."""
        if worker:
            x, y = worker.get_position()
            # check all adjacent positions for valid builds
            valid_builds = set()
            for i in range(-1,2):
                for j in range(-1,2):
                    target_position = x + i, y + j
                    if (i != 0 or j != 0) and self.can_build((x, y), target_position):
                        valid_builds.add(target_position)
            worker.set_valid_builds(valid_builds)

    # end game

    def check_height_win_condition(self, position: tuple[int,int]):
        """
        Checks if moving to the given position meets the win condition (reaching the third level).

        target_position: A tuple (x, y) indicating the position to check.
        return: True if the position is on the third level, False otherwise.
        """
        if self.get_position_height(position) == self._max_building_height:
            self._game_over = True

    def check_cannot_move_lose_condition(self, player: Player) -> bool:
        """Return True if a player cannot move. False otherwise."""
        for worker in player.get_workers():
            if worker.get_valid_moves(): # if there is at least one valid move
                return False
        self._game_over = True
        return True

    def game_over_status(self):
        """Returns True if a player has won, False otherwise."""
        return self._game_over

    # display

    def set_selected_position(self, position: tuple[int, int]) -> None:
        """Sets the current player's selected position."""
        if position is None:
            self._selected_position = None
        elif self.is_on_board(position):
            self._selected_position = position

    def get_selected_position(self) -> tuple[int, int]:
        """Gets the current player's selected position."""
        return self._selected_position

    def set_selected_worker(self, position: tuple[int, int]) -> None:
        """Selects the worker in the position.
        Unselects the worker if worker is already selected."""
        worker = self.get_position_worker(position)
        if self._selected_worker == worker:
            self._selected_worker = None
        elif worker:
            self._selected_worker = worker

    def get_selected_worker(self) -> tuple[int, int]:
        """Gets the current player's selected position."""
        return self._selected_worker

    def display_building(self, position, screen):
        """Prints the board state to the screen."""
        display_position = utils.convert_to_display_position(position)
        height = self.get_position_height(position)
        screen.blit(self._images[height], display_position)

    def display_worker(self, position, screen):
        """display worker in the specified position (if any)."""
        display_position = utils.convert_to_display_position(position)
        worker = self.get_position_worker(position)
        if worker:
            screen.blit(worker.get_player().get_piece_image(), display_position)

    def display_worker_highlight(self, position, screen):
        """Highlights the worker in the given position."""
        display_position = utils.convert_to_display_position(position)
        screen.blit(self._images['selected'], display_position)

    def display_move_hightlight(self, position, screen):
        """Highlight potential move in the given position."""
        display_position = utils.convert_to_display_position(position)
        screen.blit(self._images['move'], display_position)

    def display_build_hightlight(self, position, screen):
        """Highlight potential move in the given position."""
        display_position = utils.convert_to_display_position(position)
        screen.blit(self._images['build'], display_position)
