# pylint: disable=locally-disabled no-member, missing-module-docstring

import asyncio
import pygame
import santorini.game as gm
from santorini import utils
from santorini.config import WIDTH, HEIGHT, LEFT_CLICK

async def main():
    """Entry point to start the game."""
    # Initialize display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Santorini')

    running = True
    while running:
        # Initialize the game with the board and players
        game = gm.setup(screen)

        # Start the game
        game_loop = True
        while game_loop:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running, game_loop = False, False

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == LEFT_CLICK:
                    if game.get_game_state() == 'game_over':
                        display_position = pygame.mouse.get_pos()
                        game_loop = False
                    else:
                        display_position = pygame.mouse.get_pos()
                        position = utils.convert_to_position(display_position)
                        # Select the position and worker. Unselect worker if it is already selected.
                        game.select(position)
                        game.game_loop()

            await asyncio.sleep(0) # for pygbag to run in browser.

if __name__ == "__main__":
    asyncio.run(main())
