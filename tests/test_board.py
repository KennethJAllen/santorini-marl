"""Tests for the Board class."""

import math
import pytest

def test_grid_size(board_populated):
    grid_size = board_populated.get_grid_size()
    expected_grid_size = 5
    assert grid_size == expected_grid_size

def test_get_position_worker_populated(board_populated, worker_a2):
    position = (1, 1)
    worker = board_populated.get_position_worker(position)
    assert worker == worker_a2

def test_get_position_worker_empty(board_populated, worker_empty):
    position = (2,4)
    worker = board_populated.get_position_worker(position)
    assert worker == worker_empty

def test_set_position_worker(board_empty, worker_a1):
    position = (2,4)
    board_empty.set_position_worker(position, worker_a1)
    worker = board_empty.get_position_worker(position)
    assert worker == worker_a1

@pytest.mark.parametrize(
    "position,expected_height",
    [
        ((0,1), 2),  # position (0,1), expect height 2
        ((0,0), 0),  # position (0,0), expect height 0
        ((2,4), 0),  # position (2,4), expect height 0
        ((4,3), 3),  # position (4,3), expect height 3
        ((1,0), math.inf),  # position (1,0), expect height math.inf
    ]
)
def test_get_position_height(board_populated, position, expected_height):
    height = board_populated.get_position_height(position)
    assert height == expected_height

def test_set_position_height_finite(board_empty):
    position = (2,4)
    expected_height = 2
    board_empty.set_position_height(position, expected_height)
    height = board_empty.get_position_height(position)
    assert expected_height == height

def test_set_position_height_inf(board_empty):
    position = (2,4)
    expected_height = math.inf
    board_empty.set_position_height(position, expected_height)
    height = board_empty.get_position_height(position)
    assert expected_height == height

@pytest.mark.parametrize(
    "current_position,target_position",
    [
        ((1,1), (2,1)),  # Move worker from height 0 to height 0 orthogonal
        ((1,1), (2,0)),  # Move worker from height 0 to height 1 diagonal
        ((0,1), (0,2)),  # Move worker from height 2 to height 0
        ((0,1), (1,2)),  # Move worker from height 2 to height 3
    ]
)
def test_can_move_good(board_populated, current_position, target_position):
    value = board_populated.can_move(current_position, target_position)
    assert value is True

@pytest.mark.parametrize(
    "current_position,target_position",
    [
        ((0,0), (-1, -1)), # Move off board diagonal down-left
        ((0,0), (-1, 0)), # Move off board left
        ((0,0), (0, -1)), # Move off board down
        ((4,4), (5, 5)), # Move off board diagonal top-right
        ((4,4), (5, 4)), # Move off board top
        ((4,4), (4, 5)), # Move off board right

        ((0,0), (2,0)),  # Move a worker too far
        ((0,0), (0,0)),  # Move worker to own square

        ((0,0), (1,1)),  # Move worker ontop of another worker
        ((3,4), (3,5)),  # Move empty worker

        ((1,1), (1,2)),  # Move worker from height 1 to height 3
        ((0,0), (1,0)),  # Move worker from height 0 to height inf
    ]
)
def test_can_move_bad(board_populated, current_position, target_position):
    value = board_populated.can_move(current_position, target_position)
    assert value is False
