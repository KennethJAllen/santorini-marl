"""Game logic and main loop."""
from santorini.worker import Worker
from santorini.board import Board
import santorini.utils as utils

class Game:
    """Game logic and main loop."""

    def __init__(self, players: int, board: Board):
        self.players = players # List of Player objects participating in the game
        self.board = board  # The game board, an instance of the Board class
        self.current_player_index = 0  # Index to keep track of whose turn it is
        self.game_over = False  # Flag to check if the game has ended
        self.num_workers = 2

    def start(self):
        """Initialize game components and start the game loop"""
        self.setup_board()
        self.game_loop()

    def setup_board(self):
        """Prepare the game board for play (e.g., initialize players, place workers)"""
        for player in self.players:
            for worker_num in range(1, self.num_workers + 1):
                worker = Worker(player = player, worker_id = worker_num)
                while True:
                    print(f"Player {player.player_id} select the position for worker {worker_num}.")
                    try:
                        algebraic_worker_position = input().upper()
                        worker_position = utils.algebraic_position_to_indices(algebraic_worker_position)

                        if self.board.get_position_worker(worker_position):
                            raise ValueError("There is already a worker occupying that space. Choose a different space.")
                        self.board.set_position_worker(worker_position, worker)
                        break
                    except ValueError as e:
                        print(e)
                        continue

    def game_loop(self):
        """Main game loop"""
        while not self.game_over:
            current_player = self.players[self.current_player_index]
            self.turn(current_player)
            self.check_win_condition()

            num_players = len(self.players)
            self.current_player_index = (self.current_player_index + 1) % num_players

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
        # todo: check if position is valid
        print(f"Player {player.player_id}, select the position you want your worker to move to (e.g. B2 to move your worker to space B2)")
        target_position = input().upper()  # Assuming the input is something like 'A1'
        # todo: check if position is valid
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
