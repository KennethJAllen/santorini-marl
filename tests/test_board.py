"""Tests for the Board class."""

import math
import pytest

def test_grid_size(populated_board):
    grid_size = populated_board.get_grid_size()
    expected_grid_size = 5
    assert grid_size == expected_grid_size

def test_get_position_worker(populated_board, worker_a2):
    position = (1, 1)
    worker = populated_board.get_position_worker(position)
    assert worker == worker_a2

def test_get_position_empty_worker(populated_board, worker_empty):
    position = (2,4)
    worker = populated_board.get_position_worker(position)
    assert worker == worker_empty

def test_set_position_worker(board_empty, worker_a1):
    position = (2,4)
    board_empty.set_position_worker(position, worker_a1)
    worker = board_empty.get_position_worker(position)
    assert worker == worker_a1

@pytest.mark.parametrize(
    "position,expected_height",
    [
        ((0,1), 2),  # Test case 1: position (0,1), expect height 2
        ((0,0), 0),  # Test case 2: position (0,0), expect height 0
        ((2,4), 0),  # Test case 3: position (2,4), expect height 0
        ((4,3), 3),  # Test case 4: position (4,3), expect height 3
        ((1,0), math.inf),  # Test case 5: input (1,0), expect height math.inf
    ]
)
def test_get_position_height(populated_board, position, expected_height):
    height = populated_board.get_position_height(position)
    assert height == expected_height
