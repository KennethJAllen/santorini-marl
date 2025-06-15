"""Renders the game using Pygame."""
from pathlib import Path
import numpy as np
import pygame
from santorini.game import Game
from santorini import utils
from santorini import config

class PygameRenderer:
    def __init__(self,
                 grid_size: int = config.GRID_SIZE,
                 asset_dir: Path = Path("images/assets"),
                 screen_size=600):
        pygame.init()
        self.board_size = grid_size
        self.screen = pygame.display.set_mode((screen_size, screen_size))
        self.cell_size = screen_size // grid_size
        self.asset_dir = asset_dir
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
                self.screen, (0, 0, 0),
                (x * self.cell_size, 0),
                (x * self.cell_size, self.board_size * self.cell_size)
            )
        for y in range(self.board_size + 1):
            pygame.draw.line(
                self.screen, (0, 0, 0),
                (0, y * self.cell_size),
                (self.board_size * self.cell_size, y * self.cell_size)
            )

        # Draw buildings and workers.
        # Show the board from the perspective of the first player.
        obs = board.get_observation(0)

        for building_channel_idx in range(7):
            building_img = self.images[building_channel_idx]
            channel = obs[:,:,building_channel_idx]
            for (x, y), value in np.ndenumerate(channel):
                if value:
                    self.screen.blit(building_img, self.board_to_pixel(x, y))

        # Highlight
        for hx, hy in self.highlight_squares:
            r = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            r.fill((0, 255, 0, 100))  # semi-transparent green
            self.screen.blit(r, self.board_to_pixel(hx, hy))

        pygame.display.flip()

    def handle_mouse(self, game: Game) -> int:
        """
        Processes one click. Returns an action int if a full move+build
        has been selected, otherwise None.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = event.pos
                move = mouse_x // self.cell_size, mouse_y // self.cell_size
                gx, gy = move

                # Set up the pieces
                if game.state == game.state.SETUP:
                    space_action = utils.encode_space((gx, gy))
                    if space_action in game.valid_actions:
                        return space_action

                # Playing the game
                # 1) Pick up one of your workers
                if self.selected_worker is None:
                    starts = {utils.decode_action(a)[0] for a in game.valid_actions}
                    if move in starts:
                        self.selected_worker = move
                        # highlight legal move targets
                        self.highlight_squares = [
                            utils.decode_action(a)[1]
                            for a in game.valid_actions
                            if utils.decode_action(a)[0] == move
                        ]

                # 2) Once a worker is selected, choose **move** target
                elif self._pending_move is None and move in self.highlight_squares:
                    self._pending_move = move
                    # highlight legal build spots
                    self.highlight_squares = [
                        utils.decode_action(a)[2]
                        for a in game.valid_actions
                        if (utils.decode_action(a)[0] == self.selected_worker
                            and utils.decode_action(a)[1] == self._pending_move)
                    ]

                # 3) Now that move is set, clicking on a build‐spot completes the action
                elif self._pending_move is not None and move in self.highlight_squares:
                    for a in game.valid_actions:
                        frm, to, build = utils.decode_action(a)
                        if (frm == self.selected_worker
                                and to == self._pending_move
                                and build == move):
                            action = a
                            # reset for next turn
                            self.selected_worker = None
                            self._pending_move = None
                            self.highlight_squares = []
                            return action

        return None