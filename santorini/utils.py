"""Utility functions"""
from config import GRID_SIZE

def is_adjacent(position1: tuple[int, int], position2: tuple[int, int]) -> bool:
    """
    Returns True if position1 and position2 are horizonatally, vertically, or diagonally adjacent.
    Returns False otherwise.
    """
    x1, y1 = position1
    x2, y2 = position2
    if max(abs(x1 - x2), abs(y1 - y2)) != 1:
        # check position1 is adjacent to position2
        return False
    return True

def space_index_to_position(space_index: int, grid_size: int = GRID_SIZE) -> tuple[int, int]:
    """
    Given an integer index for the space on the board,
    each from 0 to GRID_SIZE squared (typically 25),
    return the row, col tuple representing the positon.
    Inverse function of space_position_to_index.
    """
    space_row = space_index // grid_size
    space_col = space_index % grid_size
    return space_row, space_col

def space_position_to_index(space_position: tuple[int, int], grid_size: int = GRID_SIZE) -> int:
    """
    Given a tuple of integers representing a position of a space on the board,
    each from 0 to GRID_SIZE, (typically 5),
    return the integer index of the space.
    Inverse function of space_index_to_position.
    """
    space_row, space_col = space_position
    space_index = space_row * grid_size + space_col
    return space_index
