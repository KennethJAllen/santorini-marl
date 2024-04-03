"""Test configuration file containing pytest fixtures."""

import math
from collections import defaultdict 
import pytest
from santorini.player import Player
from santorini.worker import Worker
from santorini.game import Board


@pytest.fixture
def player_a():
    """A sample worker for player a."""
    return Player(player_id = 'a')

@pytest.fixture
def player_b():
    """A sample worker for player b."""
    return Player(player_id = 'b')

@pytest.fixture
def worker_empty():
    """A derfault worker."""
    return Worker()

@pytest.fixture
def worker_a1(player_a):
    """A sample worker for player a."""
    return Worker(player = player_a, worker_id = '1')

@pytest.fixture
def worker_a2(player_a):
    """A second sample worker for player a."""
    return Worker(player = player_a, worker_id = '2')

@pytest.fixture
def worker_b1(player_b):
    """A sample worker for player b."""
    return Worker(player = player_b, worker_id = '3')

@pytest.fixture
def worker_b2(player_b):
    """A sample worker for player b."""
    return Worker(player = player_b, worker_id = '4')

@pytest.fixture
def board_empty():
    """An empty board."""
    return Board()

@pytest.fixture
def populated_board(worker_a1, worker_a2, worker_b1, worker_b2, worker_empty):
    '''A sample board.'''
    state_data = defaultdict(lambda: [Worker(), 0],
                             {(0,0): [worker_a1, 0],
                              (1,1): [worker_a2, 1],
                              (0,1): [worker_b1, 2],
                              (4,4): [worker_b2, 1],
                              (1,2): [worker_empty, 3],
                              (3,3): [worker_empty, 1],
                              (4,3): [worker_empty, 3],
                              (1,0): [worker_empty, math.inf]})
    board = Board()
    board.set_state(state_data)
    return board
