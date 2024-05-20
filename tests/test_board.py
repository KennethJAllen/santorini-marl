"""Tests for the Board class."""
# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, redefined-outer-name, missing-function-docstring

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

# get_position_height

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

# test set_position_height

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

# test is_on_board
@pytest.mark.parametrize(
    "position,expected",
    [
        ((0,0), True),
        ((2,3), True),
        ((4,4), True),
        ((0,-1), False),
        ((4,5), False)
    ]
)
def test_is_on_board(board_empty, position, expected):
    assert board_empty.is_on_board(position) == expected

# test can_move

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

# test move_worker

@pytest.mark.parametrize(
    "current_position,target_position",
    [
        ((1,1), (2,1)),  # Move worker from height 0 to height 0 orthogonal
        ((1,1), (2,0)),  # Move worker from height 0 to height 1 diagonal
        ((0,1), (0,2)),  # Move worker from height 2 to height 0
        ((0,1), (1,2)),  # Move worker from height 2 to height 3
    ]
)
def test_move_worker_good(board_populated, current_position, target_position):
    worker_in_initial_position = board_populated.get_position_worker(current_position)
    board_populated.move_worker(current_position, target_position)
    assert not board_populated.get_position_worker(current_position)
    assert worker_in_initial_position is board_populated.get_position_worker(target_position)

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
def test_move_worker_bad(board_populated, current_position, target_position):
    with pytest.raises(ValueError) as excinfo:
        board_populated.move_worker(current_position, target_position)
    assert "That is not a valid move." in str(excinfo.value)

# test can_build

def test_can_build_valid_move(board_populated):
    worker_position = (0, 1)
    build_position = (0, 2)  # Adjacent and no worker present, height not max
    assert board_populated.can_build(worker_position, build_position) is True

def test_can_build_non_adjacent_position(board_populated):
    worker_position = (0, 0)
    build_position = (2, 0)  # Not adjacent
    assert board_populated.can_build(worker_position, build_position) is False

def test_can_build_position_with_worker(board_populated):
    worker_position = (0, 0)
    build_position = (1, 1)  # Worker present
    assert board_populated.can_build(worker_position, build_position) is False

def test_can_build_max_height_reached(board_populated):
    worker_position = (0, 0)
    build_position = (1, 0)  # Max height (math.inf)
    assert board_populated.can_build(worker_position, build_position) is False

def test_can_build_off_board_position(board_populated):
    worker_position = (4, 4)
    build_position = (5, 5)  # Off board
    assert board_populated.can_build(worker_position, build_position) is False

def test_can_build_invalid_worker_position(board_populated):
    worker_position = (5, 5)  # Off board
    build_position = (0, 1)
    assert board_populated.can_build(worker_position, build_position) is False

def test_can_build_valid_position_edge_of_board(board_populated):
    worker_position = (4, 4)
    build_position = (4, 3)  # On the edge of the board
    assert board_populated.can_build(worker_position, build_position) is True

# test build

def test_build_success(board_populated):
    worker_position = (2, 0)
    build_position = (2, 1)
    
    # Expected increase in building height
    initial_height = board_populated.get_position_height(build_position)
    board_populated.build(worker_position, build_position)
    new_height = board_populated.get_position_height(build_position)

    assert new_height == initial_height + 1

def test_build_failure_dome_present(board_populated):
    worker_position = (1, 0)
    build_position = (1, 0)

    # Trying to build on a dome, where height is math.inf
    with pytest.raises(ValueError) as excinfo:
        board_populated.build(worker_position, build_position)
    assert "That is not a valid build position." in str(excinfo.value)

def test_build_failure_not_adjacent(board_populated):
    worker_position = (0, 0)
    build_position = (4, 4)

    # Trying to build on a non-adjacent position
    with pytest.raises(ValueError) as excinfo:
        board_populated.build(worker_position, build_position)
    assert "That is not a valid build position." in str(excinfo.value)

def test_build_max_height_reached(board_populated):
    worker_position = (1, 2)
    build_position = (1, 2)

    # Trying to build where the current height is already at the maximum (e.g., 3)
    with pytest.raises(ValueError) as excinfo:
        board_populated.build(worker_position, build_position)
    assert "That is not a valid build position." in str(excinfo.value)

def test_build_worker_at_position(board_populated):
    worker_position = (0, 0)
    build_position = (0, 1)

    with pytest.raises(ValueError) as excinfo:
        board_populated.build(worker_position, build_position)
    assert "That is not a valid build position." in str(excinfo.value)
