import asyncio

import pygame

from santorini.game import Game
from santorini.board import Board
from santorini.player import Player

async def main():
    """Entry point to start the game."""
    # Initialize Pygame
    pygame.init()

    # Constants for the game window
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    FPS = 30  # Frames per second

    # Set up the display
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Santorini")

    # Set up the clock
    clock = pygame.time.Clock()
    
    board = Board()
    num_players = 2 # default 2 players
    # Initialize players
    players = []
    for player_id in range(1,num_players+1):
        players.append(Player(player_id))

    # Initialize the game with the board and players
    game = Game(players, board)

    # Start the game
    running = True
    while running:
        game.setup_board()
        game.game_loop()
        await asyncio.sleep(0)

if __name__ == "__main__":
    asyncio.run(main())
