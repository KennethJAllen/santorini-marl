# pylint: disable=locally-disabled no-member, missing-module-docstring

import asyncio
import pygame
from santorini.game import Game
from santorini.board import Board
from santorini.player import Player
from santorini import utils
from santorini.config import WIDTH, HEIGHT, GRID_SIZE, FPS, NUM_PLAYERS, LEFT

# BUG: If a player loses because they have no valid moves, a click must be made before the player loses.
# TODO: Implement minimax AI to play against
# TODO: Add game over screen with restart button.
# TODO: Add piece selection highlighting and valid move highlighting

async def main():
    """Entry point to start the game."""
    # Initialize display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Santorini')
    clock = pygame.time.Clock()

    # Initialize board
    board = Board(grid_size = GRID_SIZE)
    num_players = NUM_PLAYERS

    # Initialize players
    players = []
    for player_id in range(1,num_players+1):
        players.append(Player(player_id))

    # Initialize the game with the board and players
    game = Game(players, board, screen)

    # Start the game
    running = True
    while running:
        clock.tick(FPS)

        if game.get_winner() is not None:
            print(f"Player {game.get_winner().get_player_id()} wins!")
            running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == LEFT:
                display_position = pygame.mouse.get_pos()
                position = utils.convert_to_position(display_position)
                # Select the position and worker. Unselect worker if it is already selected.
                game.select(position)
                game.game_loop()

        game.display_game()
        await asyncio.sleep(0) # for converting to WASM to run in browser.

if __name__ == "__main__":
    asyncio.run(main())
