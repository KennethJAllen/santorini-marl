"""Configuration globals."""
# display
WIDTH, HEIGHT = 320, 320 # width and height of the screen in pixels
LEFT_CLICK = 1  # integer corresponding to left click pygame event

# board parameters
GRID_SIZE = 5 # number of cells in each row and column
MAX_BUILDING_HEIGHT = 3 # Maximum height a building can reach before being capped
SQUARE_SIZE = WIDTH // GRID_SIZE # the size of each square in pixels

# game parameters
NUM_WORKERS = 2

# colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
