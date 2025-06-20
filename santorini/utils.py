"""Utility functions"""
from santorini.config import GRID_SIZE
DIRS = [
    (-1, -1),  # NW
    (0, -1),  # N
    (1, -1),  # NE
    (1, 0),  # E
    (1, 1),  # SE
    (0, 1),  # S
    (-1, 1),  # SW
    (-1, 0),  # W
    ]


def is_adjacent(position1: tuple[int, int], position2: tuple[int, int]) -> bool:
    """
    Returns True if position1 and position2 are horizonatally, vertically, or diagonally adjacent.
    Returns False otherwise.
    """
    x1, y1 = position1
    x2, y2 = position2
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    if max(dx, dy) != 1:
        # check position1 is adjacent to position2
        return False
    return True

def decode_space(space_index: int, grid_size: int = GRID_SIZE) -> tuple[int, int]:
    """
    Given an integer index for the space on the board,
    each from 0 to GRID_SIZE squared (typically 25),
    return the x, y tuple representing the positon.
    Inverse function of space_position_to_index.
    """
    y = space_index // grid_size
    x = space_index % grid_size
    return x, y

def encode_space(space_position: tuple[int, int], grid_size: int = GRID_SIZE) -> int:
    """
    Given a x, y tuple of integers representing a position of a space on the board,
    each from 0 to GRID_SIZE, (typically 5),
    return the integer index of the space.
    Inverse function of space_index_to_position.
    """
    x, y = space_position
    space_index = y * grid_size + x
    return space_index

def decode_action(action: int, grid_size: int = GRID_SIZE) -> tuple[tuple[int, int]]:
    """
    Given an action integer from 0 to grid_size * grid_size * 8 * 8 (typically 1600 when grid_size is 5),
    return the space to move from, the space to move to, and the space to build on.
    each as (x,y) coordinate tuples.

    This function can return out of bounds positions.
    """
    if not 0 <= action < grid_size * grid_size * 8 * 8:
        raise ValueError(f"Action {action} is out of bounds for grid size {grid_size}.")

    # split into (from_index, move_dir, build_dir)
    from_idx, rem = divmod(action, 8*8)      # from_idx in [0..24], rem in [0..63]
    move_dir, build_dir = divmod(rem, 8)    # each in [0..7]

    # convert linear from_idx to (x,y)
    from_x, from_y = from_idx % grid_size, from_idx // grid_size

    dx_move, dy_move   = DIRS[move_dir]
    dx_build, dy_build = DIRS[build_dir]

    to_x = from_x + dx_move
    to_y = from_y + dy_move
    build_x = to_x + dx_build
    build_y = to_y + dy_build

    move_from = from_x, from_y
    move_to = to_x, to_y
    build_on = build_x, build_y

    return move_from, move_to, build_on

def encode_action(move_build_tuple: tuple[tuple[int, int]], grid_size: int = GRID_SIZE):
    """
    Encode a move+build tuple into a single integer action.

    Parameters:
        move_build_tuple: (
            (from_x, from_y),   # coords in [0..grid_size)
            (to_x,   to_y),     # coords = from + one of 8 dir vectors
            (build_x, build_y)  # coords = to   + one of 8 dir vectors
        )

    Returns:
        action (int): in [0 .. grid_size*grid_size*8*8)

    Raises:
        ValueError if any coords are out of range [0..grid_size) for the 'from' square,
        or if the to/build steps are not one of the 8 compass directions.
    """
    (from_x, from_y), (to_x, to_y), (build_x, build_y) = move_build_tuple

    # compute move direction
    dx_move = to_x - from_x
    dy_move = to_y - from_y
    try:
        move_dir = DIRS.index((dx_move, dy_move))
    except ValueError as e:
        raise ValueError(f"Invalid move direction {(dx_move, dy_move)}; must be one of {DIRS}") from e

    # compute build direction
    dx_build = build_x - to_x
    dy_build = build_y - to_y
    try:
        build_dir = DIRS.index((dx_build, dy_build))
    except ValueError as e:
        raise ValueError(f"Invalid build direction {(dx_build, dy_build)}; must be one of {DIRS}") from e

    # linearize from-square
    from_idx = from_x + from_y * grid_size

    # pack into action integer
    action = from_idx * (8 * 8) + move_dir * 8 + build_dir
    return action

def next_player_index(player_index: int, num_players: int) -> int:
    """Returns the index of the next player, looping back to 0"""
    return (player_index + 1) % num_players

def previous_player_index(player_index: int, num_players: int) -> int:
    """Returns the index of the previous player, looping back to 0"""
    return (player_index - 1) % num_players
