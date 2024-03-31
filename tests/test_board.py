"""Tests for the Board class."""

import pytest
from santorini.game import Board

@pytest.fixture
def sample_board():
    '''A list sample card names.'''
    return Board()

def test(sample_board):
    assert sample_board.num_players == 2
