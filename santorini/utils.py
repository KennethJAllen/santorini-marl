"""Utility functions"""

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

def algebraic_position_to_indices(algebraic_position: str) -> tuple[int, int]:
    """Converts algebraic notation e.g. B3 to index notation (1, 2)"""
    if not len(algebraic_position) >= 2 or not algebraic_position[0].isalpha() or not algebraic_position[1:].isdigit():
        raise ValueError("The position must be two characters, a letter followed by a number. For example, A1.")
    first_position = ord(algebraic_position[0]) - 65 # Converts A to 0, B to 1, etc.
    second_position = int(algebraic_position[1:]) - 1 # Converts 1 to 0, 2 to 1, etc.
    return first_position, second_position
