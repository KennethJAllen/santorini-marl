"""Tests for utils.py"""

from itertools import product
import pytest
from santorini import utils


@pytest.fixture(name="grid_size")
def fixture_grid_size():
    """Example fixed grid size."""
    return 5


def test_is_adjacent():
    assert utils.is_adjacent((0, 1), (1, 1)) is True
    assert utils.is_adjacent((2, 4), (1, 3)) is True
    assert utils.is_adjacent((0, 0), (0, 1)) is True
    assert utils.is_adjacent((4, 3), (4, 3)) is False
    assert utils.is_adjacent((0, 0), (1, 2)) is False


def test_space_index_to_position(grid_size: int):
    for space_index in range(grid_size**2):
        x, y = utils.decode_space(space_index, grid_size)
        assert 0 <= x < grid_size
        assert 0 <= y < grid_size
        recovered_space_index = utils.encode_space((x, y), grid_size)
        assert recovered_space_index == space_index


def test_space_position_to_index(grid_size: int):
    for space_row in range(grid_size):
        for space_col in range(grid_size):
            space_index = utils.encode_space((space_row, space_col), grid_size)
            assert 0 <= space_index < grid_size**2
            recovered_space_row, recovered_space_col = utils.decode_space(
                space_index, grid_size
            )
            assert recovered_space_row == space_row
            assert recovered_space_col == space_col


def test_encode_decode_space(grid_size: int):
    for encoded_space in range(grid_size**2):
        decoded_space = utils.decode_space(encoded_space, grid_size)
        assert encoded_space == utils.encode_space(decoded_space, grid_size)


def test_decode_encode_soace(grid_size: int):
    for x, y in product(range(grid_size), range(grid_size)):
        encoded_space = utils.encode_space((x, y), grid_size)
        assert (x, y) == utils.decode_space(encoded_space, grid_size)


def test_encode_decode_action(grid_size: int):
    for encoded_move in range(64 * grid_size**2):
        decoded_move = utils.decode_action(encoded_move, grid_size)
        assert encoded_move == utils.encode_action(decoded_move, grid_size)


def test_decode_encode_action(grid_size):
    for x, y in product(range(grid_size), range(grid_size)):
        for move_dx, move_dy in product(range(-1, 2), range(-1, 2)):
            for build_dx, build_dy in product(range(-1, 2), range(-1, 2)):
                if (move_dx, move_dy) == (0, 0) or (build_dx, build_dy) == (0, 0):
                    continue
                move_from = (x, y)
                move_to = (x + move_dx, y + move_dy)
                build_on = (x + move_dx + build_dx, y + move_dy + build_dy)
                action = utils.encode_action((move_from, move_to, build_on), grid_size)
                decoded_move_from, decoded_move_to, decoded_build_on = (
                    utils.decode_action(action, grid_size)
                )
                assert decoded_move_from == move_from
                assert decoded_move_to == move_to
                assert decoded_build_on == build_on
