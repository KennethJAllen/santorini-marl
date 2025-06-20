"""Test configuration file containing pytest fixtures."""
# pylint: disable=protected-access

from collections import defaultdict
import pytest
from santorini.player import Player, Worker
from santorini.board import Board

@pytest.fixture(name='player_1')
def fixture_player_1():
    """A sample worker for player 1."""
    return Player(player_id = 1)

@pytest.fixture(name='player_2')
def fixture_player_2():
    """A sample worker for player 2."""
    return Player(player_id = 2)

@pytest.fixture(name='worker_empty')
def fixture_worker_empty():
    """A derfault worker."""
    return Worker()

@pytest.fixture(name='worker_a1')
def fixture_worker_a1(player_1):
    """A sample worker for player a."""
    return Worker(player = player_1, worker_id = 1)

@pytest.fixture(name='worker_a2')
def fixture_worker_a2(player_1):
    """A second sample worker for player a."""
    return Worker(player = player_1, worker_id = 2)

@pytest.fixture(name='worker_b1')
def fixture_worker_b1(player_2):
    """A sample worker for player b."""
    return Worker(player = player_2, worker_id = 1)

@pytest.fixture(name='worker_b2')
def fixture_worker_b2(player_2):
    """A sample worker for player b."""
    return Worker(player = player_2, worker_id = 2)

@pytest.fixture(name='board_empty')
def fixture_board_empty():
    """An empty board."""
    return Board()


@pytest.fixture(name='board_populated')
def fixture_board_populated(worker_a1, worker_a2, worker_b1, worker_b2, worker_empty):
    """
    A sample board.
    state data key: tuple of integers (x,y) representing location on the board
    state data value: list of worker and building height.
    """
    state_data = defaultdict(lambda: [Worker(), 0],
                             {(0,0): [worker_a1, 0],
                              (1,1): [worker_a2, 0],
                              (0,1): [worker_b1, 2],
                              (4,4): [worker_b2, 1],
                              (2,0): [worker_empty, 1],
                              (3,3): [worker_empty, 1],
                              (1,2): [worker_empty, 3],
                              (4,3): [worker_empty, 3],
                              (1,0): [worker_empty, 4]})
    board = Board()
    board._state = state_data
    return board
