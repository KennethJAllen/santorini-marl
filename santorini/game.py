"""Game class containing game logic."""
from santorini.board import Board
from santorini.player import Player, Worker
import santorini.utils as utils

# todo: make it so if a player has no moves they lose

class Game:
    """Game logic, setup, and main loop."""

    def __init__(self, players: list[Player], board: Board):
        self.players = players # List of Player objects participating in the game
        self.board = board  # The game board, an instance of the Board class
        self.num_workers = 2 # number of workers each player has
        self.current_player_index = 0  # Index to keep track of whose turn it is

    def start(self):
        """Initialize game components and start the game loop"""
        self.setup_board()
        self.game_loop()

    def setup_board(self):
        """Prepare the game board for play (e.g., initialize players, place workers)"""
        for player in self.players:
            for worker_num in range(1, self.num_workers + 1):
                worker = Worker(player = player, worker_id = worker_num)
                player.add_worker(worker) # add worker to player's list of workers
                self.board.display()
                while True:
                    print(f"Player {player.player_id} select the position for worker {worker_num}.")
                    try:
                        algebraic_worker_position = input().upper()
                        worker_position = utils.algebraic_position_to_indices(algebraic_worker_position)
                        self.board.set_position_worker(worker_position, worker)
                        break
                    except ValueError as error:
                        print(error)
                        continue

    def game_loop(self):
        """Main game loop."""
        while True:
            current_player = self.players[self.current_player_index]
            # player's move action
            worker_position, target_position = self.get_move_action(current_player)
            self.execute_move(worker_position, target_position)
            # check if move won the game
            if self.check_win_condition():
                break
            # player's build action
            build_position = self.get_build_action(target_position, current_player)
            self.execute_build(target_position, build_position)
            # next player's turn
            num_players = len(self.players)
            self.current_player_index = (self.current_player_index + 1) % num_players
        self.end_game(current_player)

    def get_move_action(self, player: Player) -> tuple[tuple[int, int], tuple[int, int]]:
        """Get the player's choice of move action.
        Returns position to move worker from and location to move worker to."""
        self.board.display()
        worker_position = self.get_position_move_from(player)
        target_position = self.get_position_move_to(player, worker_position)
        return worker_position, target_position

    def get_position_move_from(self, player: Player) -> tuple[int,int]:
        """Get the position to move a worker from."""
        while True:
            try:
                print(f"Player {player.player_id}, select the position you want to move from (e.g. A1 to move worker on space A1)")
                algebraic_worker_position = input().upper()  # Assuming the input is something like 'A1'
                worker_position = utils.algebraic_position_to_indices(algebraic_worker_position)
                self.validate_correct_player(worker_position, player)
                break
            except ValueError as error:
                print(error)
                continue
        return worker_position

    def get_position_move_to(self, player: Player, worker_position: tuple[int, int]) -> tuple[int,int]:
        """Get the position to move a worker to."""
        while True:
            try:
                print(f"Player {player.player_id}, select the position you want to move to (e.g. B2 to move your worker to space B2)")
                algebraic_target_position = input().upper()  # Assuming the input is something like 'A1'
                target_position = utils.algebraic_position_to_indices(algebraic_target_position)
                self.validate_move(worker_position, target_position)
                break
            except ValueError as error:
                print(error)
                continue
        return target_position

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
                self.validate_build_position(worker_position, build_position)
                break
            except ValueError as error:
                print(error)
        return build_position

    def execute_build(self, worker_position, build_position):
        """Execute the build action on the board."""
        self.board.build(worker_position, build_position)

    def check_win_condition(self):
        """Check if a player has won the game"""
        return self.board.game_over_status()

    def end_game(self, player):
        """Perform any cleanup and declare the game winner."""
        print(f"Player {player.player_id} wins!")

    def validate_worker_placement(self, position):
        """Verifies that a worker can be placed on position."""
        if not self.board.can_place(position):
            raise ValueError("Cannot place a worker on that position.")

    def validate_move(self, worker_position, target_position):
        """Verifies that a worker in worker position can move to target position."""
        if not self.board.can_move(worker_position, target_position):
            raise ValueError("Cannot move to that position.")

    def validate_correct_player(self, position: tuple[int, int], player: Player):
        """Verifies that the worker in the specified position belongs to player."""
        # validate position on board
        if not self.board.is_on_board(position):
            grid_size = self.board.get_grid_size()
            raise ValueError(f"Choose a position on the {grid_size} by {grid_size} board.")
        # validate player
        if not self.board.get_position_worker(position).get_player() is player:
            raise ValueError(f"There is no worker in this position, or it does not belong to player {player.player_id}.")

    def validate_build_position(self, worker_position: tuple[int, int], build_position: tuple[int, int]):
        """Verifies that a position can be built on."""
        if not self.board.can_build(worker_position, build_position):
            raise ValueError("That is not a valid build position. Please choose a valid build position.")
