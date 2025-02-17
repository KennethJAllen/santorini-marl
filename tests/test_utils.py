"""Tests for utils.py"""
import pytest
from santorini import utils

@pytest.fixture(name='grid_size')
def fixture_grid_size():
    """Example fixed grid size."""
    return 5

def test_is_adjacent():
    assert utils.is_adjacent((0,1), (1,1)) is True
    assert utils.is_adjacent((2,4), (1,3)) is True
    assert utils.is_adjacent((0,0), (0,1)) is True
    assert utils.is_adjacent((4,3), (4,3)) is False
    assert utils.is_adjacent((0,0), (1,2)) is False

def test_space_index_to_position(grid_size):
    for space_index in range(grid_size**2):
        space_row, space_col = utils.space_index_to_position(space_index, grid_size)
        assert 0 <= space_row < grid_size
        assert 0 <= space_col < grid_size
        recovered_space_index = utils.space_position_to_index((space_row, space_col), grid_size)
        assert recovered_space_index == space_index

def test_space_position_to_index(grid_size):
    for space_row in range(grid_size):
        for space_col in range(grid_size):
            space_index = utils.space_position_to_index((space_row, space_col), grid_size)
            assert 0 <= space_index < grid_size**2
            recovered_space_row, recovered_space_col = utils.space_index_to_position(space_index, grid_size)
            assert recovered_space_row == space_row
            assert recovered_space_col == space_col
