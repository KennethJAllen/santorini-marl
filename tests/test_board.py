"""Tests for the Board class."""

import pytest
from santorini.worker import Worker
from santorini.game import Board

@pytest.fixture
def worker_a1():
    """A sample worker for player a."""
    return Worker(player = 'a', worker_id = '1')

@pytest.fixture
def worker_a2():
    """A second sample worker for player a."""
    return Worker(player = 'a', worker_id = '2')

@pytest.fixture
def worker_b1():
    """A sample worker for player b."""
    return Worker(player = 'b', worker_id = '3')

@pytest.fixture
def board():
    '''A sample board.'''
    return Board()

def test(sample_board):
    assert sample_board.num_players == 2
