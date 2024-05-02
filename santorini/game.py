"""Game logic and main loop."""
from collections import defaultdict
from santorini.player import Player
from santorini.worker import Worker
from santorini.board import Board

class GameState:
    """Game logic and main loop."""

    def __init__(self, players: list[Player], board: Board):
        self.players = players  # List of Player objects participating in the game
        self.board = board  # The game board, an instance of the Board class
        self.current_player_index = 0  # Index to keep track of whose turn it is
        self.game_over = False  # Flag to check if the game has ended

    def start(self):
        """Initialize game components and start the game loop"""
        self.setup_board()
        self.game_loop()

    def setup_board(self):
        """Prepare the game board for play (e.g., place workers)"""

    def game_loop(self):
        """Main game loop"""
        while not self.game_over:
            current_player = self.players[self.current_player_index]
            self.turn(current_player)
            self.check_win_condition()
            self.current_player_index = (self.current_player_index + 1) % len(self.players)

    def turn(self, player):
        """
        Handle a single player's turn
        This involves moving a worker and then performing a build action
        """
        move_action = self.get_move_action(player)
        self.execute_move(move_action)
        #build_action = self.get_build_action(player)
        #self.execute_build(build_action)

    def get_move_action(self, player) -> dict[str, str]:
        """
        Get the player's choice of move action
        This would involve selecting a worker and a destination
        """
        self.board.display()
        print(f"Player {player.player_id}, select the position you want your worker to move from (e.g. A1 to move worker on space A1)")
        worker_position = input().upper()  # Assuming the input is something like 'A1'
        print(f"Player {player.player_id}, select the position you want your worker to move to (e.g. B2 to move your worker to space B2)")
        target_position = input().upper()  # Assuming the input is something like 'A1'
        return {'worker': worker_position, 'target': target_position}

    def execute_move(self, action):
        """
        Execute the move action on the board
        """

    def get_build_action(self, player):
        """
        Get the player's choice of build action
        This would involve selecting a building location adjacent to the worker
        """

    def execute_build(self, action):
        """Execute the build action on the board."""

    def check_win_condition(self):
        """Check if a player has won the game"""

    def end_game(self):
        """Perform any cleanup and declare the game winner."""
        self.game_over = True

