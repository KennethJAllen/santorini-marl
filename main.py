# pylint: disable=locally-disabled no-member, missing-module-docstring

import asyncio
import pygame

from santorini.game import Game
from santorini.board import Board
from santorini.player import Player
from santorini.config import WIDTH, HEIGHT, GRID_SIZE, FPS, NUM_PLAYERS

async def main():
    """Entry point to start the game."""
    # Set up the display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Santorini')
    clock = pygame.time.Clock()

    board = Board(grid_size = GRID_SIZE)
    num_players = NUM_PLAYERS # default 2 players

    # Initialize players
    players = []
    for player_id in range(1,num_players+1):
        players.append(Player(player_id))

    # Initialize the game with the board and players
    game = Game(players, board)

    # Start the game
    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                pass
        #game.setup_board()
        #game.game_loop()
        board.display(screen)
        pygame.display.update()
        await asyncio.sleep(0)

if __name__ == "__main__":
    asyncio.run(main())
