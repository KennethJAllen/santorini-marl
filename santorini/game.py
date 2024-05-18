"""Game logic and main loop."""
from santorini.board import Board
from santorini.player import Player, Worker
import santorini.utils as utils

class Game:
    """Game logic and main loop."""

    def __init__(self, players: int, board: Board):
        self.players = players # List of Player objects participating in the game
        self.board = board  # The game board, an instance of the Board class
        self.num_workers = 2 # number of workers each player has
        self.current_player_index = 0  # Index to keep track of whose turn it is
        self._game_over = False  # Flag to check if the game has ended

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
                        self.validate_position(worker_position)
                        self.board.set_position_worker(worker_position, worker)
                        break
                    except ValueError as error:
                        print(error)
                        continue

    def game_loop(self):
        """Main game loop"""
        while not self.check_win_condition():
            current_player = self.players[self.current_player_index]
            self.turn(current_player)
            num_players = len(self.players)
            self.current_player_index = (self.current_player_index + 1) % num_players
        print(f"Player {current_player.player_id} wins!")

    def turn(self, player):
        """Handle a single player's turn. This involves moving a worker and then performing a build action."""
        worker_position, target_position = self.get_move_action(player)
        self.execute_move(worker_position, target_position)
        build_position = self.get_build_action(target_position, player)
        self.execute_build(build_position, target_position)

    def get_move_action(self, player) -> tuple[tuple[int, int], tuple[int, int]]:
        """Get the player's choice of move action.
        Returns position to move worker from and location to move worker to."""
        self.board.display()
        while True:
            try:
                print(f"Player {player.player_id}, select the position you want your worker to move from (e.g. A1 to move worker on space A1)")
                algebraic_worker_position = input().upper()  # Assuming the input is something like 'A1'
                worker_position = utils.algebraic_position_to_indices(algebraic_worker_position)
                self.validate_position_on_board(worker_position)
                self.validate_correct_player(worker_position, player)
                print(f"Player {player.player_id}, select the position you want your worker to move to (e.g. B2 to move your worker to space B2)")
                algebraic_target_position = input().upper()  # Assuming the input is something like 'A1'
                target_position = utils.algebraic_position_to_indices(algebraic_target_position)
                self.validate_position(target_position, adjacent_position=worker_position)
                break
            except ValueError as error:
                print(error)
                continue
        return worker_position, target_position

    def execute_move(self, worker_position, target_position):
        """
        Moves the worker in worker_position to target_position.
        """
        self.board.move_worker(worker_position, target_position)

    def get_build_action(self, worker_position, player):
        """
        Get the player's choice of build action
        This would involve selecting a building location adjacent to the worker
        """
        self.board.display()
        while True:
            try:
                print(f"Player {player.player_id}, select the position you want to build on (e.g. A1 to build on space A1)")
                algebraic_build_position = input().upper()  # Assuming the input is something like 'A1'
                build_position = utils.algebraic_position_to_indices(algebraic_build_position)
                self.validate_position(build_position, adjacent_position=worker_position)
                break
            except ValueError as error:
                print(error)
        return build_position

    def execute_build(self, build_location, worker_position):
        """Execute the build action on the board."""
        self.board.build(build_location, worker_position)

    def validate_position_on_board(self, position):
        """Verifies that position is on the board."""
        if not self.board.is_on_board(position):
            grid_size = self.board.get_grid_size()
            raise ValueError(f"Choose a position on the {grid_size} by {grid_size} board.")

    def validate_position(self, position: tuple[int, int], adjacent_position: tuple[int, int] = None):
        """Verifies that the position is on the board and does not contain a worker.
        optional adjacent_position: Verifies that the position is adjacent to the adjacent position."""
        self.validate_position_on_board(position)
        if adjacent_position and not utils.is_adjacent(position, adjacent_position):
            raise ValueError("The target position must be adjacent to the worker's position.")
        if self.board.get_position_worker(position):
            raise ValueError("There is a worker occupying that space. Choose a different space.")

    def validate_correct_player(self, position: tuple[int, int], player: Player):
        """Verifies that the worker in the specified position belongs to player."""
        if not self.board.get_position_worker(position).get_player() == player:
            raise ValueError(f"The worker in this position does not belong to player {player.player_id}.")

    def check_win_condition(self):
        """Check if a player has won the game"""
        return self._game_over

    def end_game(self):
        """Perform any cleanup and declare the game winner."""
        self._game_over = True
