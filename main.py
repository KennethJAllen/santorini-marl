from santorini.game import Game
from santorini.board import Board
from santorini.player import Player

def main():
    """Entry point to start the game."""
    board = Board()

    # Initialize players, default 2 players
    players = [Player(1), Player(2)]

    # Initialize the game with the board and players
    game = Game(players, board)

    # Start the game
    game.start()

if __name__ == "__main__":
    main()
