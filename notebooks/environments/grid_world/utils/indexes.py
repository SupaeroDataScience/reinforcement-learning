from enum import Enum


class Direction(Enum):
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3


class TileType(Enum):
    EMPTY = 0
    WALL = 1
    START = 3
    TERMINAL = 4


class Colors(Enum):
    EMPTY = [250, 250, 250]
    WALL = [50, 54, 51]
    START = [213, 219, 214]
    TERMINAL = [73, 179, 101]
    TILE_BORDER = [50, 54, 51]
    AGENT = [219, 0, 0]
    GOAL = [0, 255, 0]
