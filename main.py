# pylint: disable=locally-disabled no-member, missing-module-docstring

import asyncio
import pygame
from santorini.game import Game
from santorini.config import WIDTH, HEIGHT, LEFT_CLICK

async def main():
    """Entry point to start the game."""
    # Initialize display
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Santorini')

    # Initialize the game
    game = Game(screen)
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
