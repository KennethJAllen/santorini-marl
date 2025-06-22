"""Tests for player.py"""
# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, redefined-outer-name, missing-function-docstring

def test_player_add_worker(player_1, worker_a1, worker_a2):
    player_1.add_worker(worker_a1)
    player_1.add_worker(worker_a2)
    result = player_1.workers
    expected = [worker_a1, worker_a2]
    assert result == expected
