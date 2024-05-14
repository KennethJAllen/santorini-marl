"""Utility functions"""

def is_adjacent(current_position: tuple[int, int], target_position: tuple[int, int]) -> bool:
    """
    Returns True if position1 and position2 are horizonatally, vertically, or diagonally adjacent.
    Returns False otherwise.
    """
    x_current, y_current = current_position
    x_target, y_target = target_position
    if max(abs(x_current - x_target), abs(y_current - y_target)) != 1:
        # check target_position is adjacent to current_position
        return False
    return True
