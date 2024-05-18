from santorini.game import Game
from santorini.board import Board
from santorini.player import Player

def main():
    """Entry point to start the game."""
    board = Board()
    num_players = 2 # default 2 players
    # Initialize players
    players = []
    for player_id in range(1,num_players+1):
        players.append(Player(player_id))

    # Initialize the game with the board and players
    game = Game(players, board)

    # Start the game
    game.start()

if __name__ == "__main__":
    main()
