"""Tests for the Board class."""
# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, redefined-outer-name, missing-function-docstring, protected-access

import pytest
from santorini.board import Board
from santorini.player import Worker


def test_grid_size(board_populated: Board):
    grid_size = board_populated.grid_size
    expected_grid_size = 5
    assert grid_size == expected_grid_size


def test_get_position_worker_populated(board_populated: Board, worker_a2: Worker):
    position = (1, 1)
    worker = board_populated.get_worker(position)
    assert worker == worker_a2


def test_get_position_worker_empty(board_populated: Board, worker_empty: Worker):
    position = (2, 4)
    worker = board_populated.get_worker(position)
    assert worker == worker_empty


def test_set_position_worker(board_empty: Board, worker_a1: Worker):
    position = (2, 4)
    board_empty._set_position_worker(position, worker_a1)
    worker = board_empty.get_worker(position)
    assert worker == worker_a1


# _get_position_height


@pytest.mark.parametrize(
    "position,expected_height",
    [
        ((0, 1), 2),  # position (0,1), expect height 2
        ((0, 0), 0),  # position (0,0), expect height 0
        ((2, 4), 0),  # position (2,4), expect height 0
        ((4, 3), 3),  # position (4,3), expect height 3
        ((1, 0), 4),  # position (1,0), expect height 4
    ],
)
def test_get_height(
    board_populated: Board, position: tuple[int, int], expected_height: int
):
    height = board_populated.get_height(position)
    assert height == expected_height


# test is_on_board
@pytest.mark.parametrize(
    "position,expected",
    [((0, 0), True), ((2, 3), True), ((4, 4), True), ((0, -1), False), ((4, 5), False)],
)
def test_is_on_board(board_empty: Board, position: tuple[int, int], expected: bool):
    assert board_empty._is_on_board(position) == expected


# test can_move


@pytest.mark.parametrize(
    "current_position,target_position",
    [
        ((1, 1), (2, 1)),  # Move worker from height 0 to height 0 orthogonal
        ((1, 1), (2, 0)),  # Move worker from height 0 to height 1 diagonal
        ((0, 1), (0, 2)),  # Move worker from height 2 to height 0
        ((0, 1), (1, 2)),  # Move worker from height 2 to height 3
    ],
)
def test_can_move_good(
    board_populated: Board,
    current_position: tuple[int, int],
    target_position: tuple[int, int],
):
    value = board_populated._can_move(current_position, target_position)
    assert value is True


@pytest.mark.parametrize(
    "current_position,target_position",
    [
        ((0, 0), (-1, -1)),  # Move off board diagonal down-left
        ((0, 0), (-1, 0)),  # Move off board left
        ((0, 0), (0, -1)),  # Move off board down
        ((4, 4), (5, 5)),  # Move off board diagonal top-right
        ((4, 4), (5, 4)),  # Move off board top
        ((4, 4), (4, 5)),  # Move off board right
        ((0, 0), (2, 0)),  # Move a worker too far
        ((0, 0), (0, 0)),  # Move worker to own square
        ((0, 0), (1, 1)),  # Move worker ontop of another worker
        ((3, 4), (3, 5)),  # Move empty worker
        ((1, 1), (1, 2)),  # Move worker from height 1 to height 3
        ((0, 0), (1, 0)),  # Move worker from height 0 to height inf
    ],
)
def test_can_move_bad(
    board_populated: Board,
    current_position: tuple[int, int],
    target_position: tuple[int, int],
):
    value = board_populated._can_move(current_position, target_position)
    assert value is False


# test move_worker


@pytest.mark.parametrize(
    "current_position,target_position",
    [
        ((1, 1), (2, 1)),  # Move worker from height 0 to height 0 orthogonal
        ((1, 1), (2, 0)),  # Move worker from height 0 to height 1 diagonal
        ((0, 1), (0, 2)),  # Move worker from height 2 to height 0
        ((0, 1), (1, 2)),  # Move worker from height 2 to height 3
    ],
)
def test_move_worker(
    board_populated: Board,
    current_position: tuple[int, int],
    target_position: tuple[int, int],
):
    worker_in_initial_position = board_populated.get_worker(current_position)
    board_populated.move_worker(current_position, target_position)
    assert not board_populated.get_worker(current_position)
    assert worker_in_initial_position is board_populated.get_worker(target_position)


# test build


def test_build_success(board_populated: Board):
    build_position = (2, 1)
    initial_height = board_populated.get_height(build_position)
    board_populated.build(build_position)
    new_height = board_populated.get_height(build_position)

    assert new_height == initial_height + 1


def test_build_failure_dome_present(board_populated: Board):
    """Trying to build on a dome, where height is 4"""
    build_position = (1, 0)
    with pytest.raises(ValueError) as excinfo:
        board_populated.build(build_position)
    assert "That is not a valid build position." in str(excinfo.value)
