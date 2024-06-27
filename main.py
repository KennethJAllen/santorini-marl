# pylint: disable=locally-disabled no-member, missing-module-docstring

import asyncio
import pygame
from santorini.game import Game
from santorini.board import Board
from santorini.player import Player
from santorini.config import WIDTH, HEIGHT, LEFT_CLICK, GRID_SIZE, NUM_PLAYERS

async def main():
    """Entry point to start the game."""
    # Initialize display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Santorini')

    board = Board(grid_size = GRID_SIZE)

    # Initialize players
    players = []
    for player_id in range(1,NUM_PLAYERS+1):
        players.append(Player(player_id))
    if NUM_PLAYERS == 1:
        players.append(Player(2, ai = True))

    # Initialize the game with the board and players
    game = Game(players, board, screen)
    game.display_game()

    # Start the game
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == LEFT_CLICK:
                # Main loop
                display_position = pygame.mouse.get_pos()
                game.game_loop(display_position)
                game.display_game()
        await asyncio.sleep(0) # for pygbag to run in browser.

if __name__ == "__main__":
    asyncio.run(main())
