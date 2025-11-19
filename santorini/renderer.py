"""Renders the game using Pygame."""

from pathlib import Path
import numpy as np
import pygame
from santorini.game import Game
from santorini import utils
from santorini import config


class PygameRenderer:
    def __init__(
        self,
        grid_size: int = config.GRID_SIZE,
        asset_dir: Path = Path("images/assets"),
        screen_size=600,
    ):
        pygame.init()
        self.board_size = grid_size
        self.screen = pygame.display.set_mode((screen_size, screen_size))
        self.cell_size = screen_size // grid_size
        self.asset_dir = asset_dir
        self.clock = pygame.time.Clock()
        self.load_assets()
        self.selected_worker = None
        self.highlight_squares = []
        self._pending_move = None

    def load_assets(self):
        self.images = {
            0: pygame.image.load(f"{self.asset_dir}/level0.png"),
            1: pygame.image.load(f"{self.asset_dir}/level1.png"),
            2: pygame.image.load(f"{self.asset_dir}/level2.png"),
            3: pygame.image.load(f"{self.asset_dir}/level3.png"),
            4: pygame.image.load(f"{self.asset_dir}/dome.png"),
            5: pygame.image.load(f"{self.asset_dir}/player1.png"),
            6: pygame.image.load(f"{self.asset_dir}/player2.png"),
            7: pygame.image.load(f"{self.asset_dir}/player3.png"),
        }
        # scale to cell_size
        for d in self.images:
            self.images[d] = pygame.transform.smoothscale(
                self.images[d], (self.cell_size, self.cell_size)
            )

    def board_to_pixel(self, x, y):
        return x * self.cell_size, y * self.cell_size

    def draw(self, game: Game):
        self.screen.fill((255, 255, 255))
        board = game.board

        # Draw grid
        for x in range(self.board_size + 1):
            pygame.draw.line(
                self.screen,
                (0, 0, 0),
                (x * self.cell_size, 0),
                (x * self.cell_size, self.board_size * self.cell_size),
            )
        for y in range(self.board_size + 1):
            pygame.draw.line(
                self.screen,
                (0, 0, 0),
                (0, y * self.cell_size),
                (self.board_size * self.cell_size, y * self.cell_size),
            )

        # Draw buildings and workers.
        # Show the board from the perspective of the first player.
        obs = board.get_observation(0)

        # Draw buildings first (channels 0-4: empty, height1, height2, height3, dome)
        for channel_idx in range(5):  # 0-4 are building channels
            channel = obs[:, :, channel_idx]
            for (x, y), value in np.ndenumerate(channel):
                if not value:
                    continue
                # Draw building from ground up
                for idx in range(channel_idx + 1):
                    building_img = self.images[idx]
                    self.screen.blit(building_img, self.board_to_pixel(x, y))

        # Draw workers (channels 5-8: current player workers 0-1, opponent workers 0-1)
        # Map observation channels 5-8 to player images 5-7 (we only have 3 player images)
        worker_channel_to_image = {
            5: 5,  # current player worker 0 -> player1 image
            6: 5,  # current player worker 1 -> player1 image
            7: 6,  # opponent worker 0 -> player2 image
            8: 6,  # opponent worker 1 -> player2 image
        }

        for channel_idx in range(5, 9):  # channels 5-8 are individual workers
            channel = obs[:, :, channel_idx]
            for (x, y), value in np.ndenumerate(channel):
                if not value:
                    continue
                worker_img = self.images[worker_channel_to_image[channel_idx]]
                self.screen.blit(worker_img, self.board_to_pixel(x, y))

        # Highlight
        for hx, hy in self.highlight_squares:
            r = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            r.fill((0, 255, 0, 100))  # semi-transparent green
            self.screen.blit(r, self.board_to_pixel(hx, hy))

        pygame.display.flip()

    def tick(self, game: Game):
        # Pump events & allow quitting
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
        # Draw & flip
        self.draw(game)
        pygame.display.flip()
        # Cap framerate
        self.clock.tick(30)

    def get_human_action(self, game: Game) -> int:
        # Wait until a valid move+build action is clicked
        while True:
            # Pump quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    move = mx // self.cell_size, my // self.cell_size
                    # handle setup or play clicks
                    action = self._process_click(game, move)
                    if action is not None:
                        return action
            # redraw highlights while waiting
            self.tick(game)

    def _process_click(self, game: Game, move: tuple) -> int:
        # Returns action int if valid, else None
        gx, gy = move
        if game.state == game.state.SETUP:
            act = utils.encode_space((gx, gy))
            return act if act in game.valid_actions else None

        # picking worker
        if self.selected_worker is None:
            starts = {utils.decode_action(a)[0] for a in game.valid_actions}
            if move in starts:
                self.selected_worker = move
                self.highlight_squares = [
                    utils.decode_action(a)[1]
                    for a in game.valid_actions
                    if utils.decode_action(a)[0] == move
                ]
            return None

        # picking move target
        if self._pending_move is None and move in self.highlight_squares:
            self._pending_move = move
            self.highlight_squares = [
                utils.decode_action(a)[2]
                for a in game.valid_actions
                if (
                    utils.decode_action(a)[0] == self.selected_worker
                    and utils.decode_action(a)[1] == self._pending_move
                )
            ]
            return None

        # picking build target
        if self._pending_move is not None and move in self.highlight_squares:
            for a in game.valid_actions:
                frm, to, build = utils.decode_action(a)
                if (
                    frm == self.selected_worker
                    and to == self._pending_move
                    and build == move
                ):
                    action = a
                    # reset
                    self.selected_worker = None
                    self._pending_move = None
                    self.highlight_squares = []
                    return action
        return None
