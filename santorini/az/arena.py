"""Pit two agents against each other on fastgame states."""

from __future__ import annotations

from dataclasses import dataclass

from santorini.az import fastgame

# Every non-winning move adds one build level and total levels are capped at
# 25*4, so real games end well under this; exceeding it means an engine bug.
MAX_PLIES = 300


def play_game(first, second) -> int:
    """Play one game; returns 0 if `first` (the seat-0 agent) wins, 1 if
    `second` wins. Agents are reset before play."""
    first.reset()
    second.reset()
    agents = (first, second)
    state = fastgame.initial_state()
    for _ in range(MAX_PLIES):
        state = fastgame.step(state, agents[state.player].act(state))
        if fastgame.is_terminal(state):
            return state.winner
    raise RuntimeError(f"Game exceeded {MAX_PLIES} plies")


@dataclass
class MatchResult:
    wins_a: int
    wins_b: int
    seat0_wins: int
    games: int

    @property
    def winrate_a(self) -> float:
        return self.wins_a / self.games if self.games else 0.0

    @property
    def seat0_fraction(self) -> float:
        return self.seat0_wins / self.games if self.games else 0.0


def play_match(agent_a, agent_b, games: int) -> MatchResult:
    """Play `games` games with agent_a taking seat 0 in even games."""
    wins_a = seat0_wins = 0
    for i in range(games):
        if i % 2 == 0:
            winner = play_game(agent_a, agent_b)
            a_won = winner == 0
        else:
            winner = play_game(agent_b, agent_a)
            a_won = winner == 1
        wins_a += a_won
        seat0_wins += winner == 0
    return MatchResult(wins_a, games - wins_a, seat0_wins, games)
