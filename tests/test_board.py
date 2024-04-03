"""Tests for the Board class."""

def test_grid_size(board_1):
    assert board_1.get_grid_size() == 5

def test_get_position_worker(board_1, worker_a2):
    position = (1, 1)
    assert board_1.get_position_worker(position) == worker_a2

def test_set_position_worker(board_2, worker_a1):
    position = (2,4)
    board_2.set_position_worker(position, worker_a1)
    assert board_2.get_position_worker(position) == worker_a1
