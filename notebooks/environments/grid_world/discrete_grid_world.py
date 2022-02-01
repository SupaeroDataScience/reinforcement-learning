import math
import os
from queue import PriorityQueue
from typing import Any, Tuple, Union, Dict, Optional
import gym
from gym import spaces
import numpy as np
from PIL import Image
import pathlib

from matplotlib import pyplot as plt

from environments.grid_world import settings
from environments.grid_world.utils.indexes import *


# un environment custom simple
def euclidean_distance(coordinates_1: tuple, coordinates_2: tuple) -> float:
    x1, y1 = coordinates_1
    x2, y2 = coordinates_2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


class DiscreteGridWorld(gym.Env):
    start_coordinates: Tuple[Union[int, Any], int]
    metadata = {'render.modes': ['human']}

    def __init__(self, map_id=settings.map_id):
        self.grid = []
        self.start_coordinates = (0, 0)
        self.agent_coordinates = None
        self.width = None
        self.height = None
        map_path = str(pathlib.Path().absolute())
        path = ""
        if "/" in map_path:
            separator = "/"
        elif "\\" in map_path:
            separator = "\\"
        else:
            raise Exception("No separator found in path: ", map_path)
        path = os.getcwd()
        path += "/environments/grid_world/maps/"
        self.load_map(path + "map" + str(map_id) + ".txt")
        # self.load_map("implem/environments/grid_world/map1.txt")
        self.observation_space = spaces.Box(-1, 1, (2,))
        self.action_space = spaces.Discrete(len(Direction))
        self.possibleActions = Direction
        self.time_step_id = 0

        # Window to use for human rendering mode
        self.window = None

        self.reset()
       
    def reset_with_map_id(self, map_id=1):
        self.__init__(map_id=map_id)

    def load_map(self, map_file):
        """
        Load the map set in environment settings.
        :param map_file: path to map description file
        """
        file = open(map_file, "r")
        self.grid = []
        y = 0
        x, start_x, start_y = None, None, None
        for line in file:
            if line[0] == '#':
                continue
            row = []
            x = 0
            for elt in line.rstrip():
                elt = int(elt)
                row.append(elt)
                if elt == TileType.START.value:
                    self.start_coordinates = (x, y)
                x += 1
            self.grid.append(row)
            y += 1
        if not x:
            raise FileExistsError("Map file is empty.")
        self.width = x
        self.height = y

        for line in self.grid:
            assert len(self.grid[0]) == len(line)
        return self.grid

    def get_state(self, x, y):
        """
        Return a numpy array (state) that belongs to X and Y coordinates in the grid.
        """
        x_value = x / self.width
        y_value = y / self.height
        return np.asarray([x_value, y_value])

    def get_coordinates(self, state):
        """
        Convert a state (list of float between 0 and 1 that represent agent's coordinate) into real agent coordinate
        inside the grid (two integers)
        """
        return int(state[0].item() * self.width), int(state[1].item() * self.height)

    def get_tile_type(self, x, y):
        return TileType(self.grid[y][x])

    def is_terminal_tile(self, x, y):
        """
        A terminal tile is a tile (aka. an environment position, wall or reachable state) that end the episode.
        In this case, it's a tile that contain a reward.
        """
        state_type = self.get_tile_type(x, y)
        return state_type == TileType.TERMINAL

    def is_available(self, x, y):
        """
        Return True is the coordinate x, y is available, aka. if it's not a wall coordinate or if this tile is not
        outside the environment. Return False otherwise.
        """
        # False for 218, 138
        # if we move into a row not in the grid
        if 0 > x or x >= self.width or 0 > y or y >= self.height:
            return False
        if self.get_tile_type(x, y) == TileType.WALL:
            return False
        return True

    def get_new_coordinates(self, action):
        """
        Return the new agent coordinates after he took the given actions.
        """
        agent_x, agent_y = self.agent_coordinates
        if Direction(action) == Direction.TOP:
            agent_y -= 1
        elif Direction(action) == Direction.BOTTOM:
            agent_y += 1
        elif Direction(action) == Direction.LEFT:
            agent_x -= 1
        elif Direction(action) == Direction.RIGHT:
            agent_x += 1
        else:
            raise AttributeError("Unknown action")
        return agent_x, agent_y

    def step(self, action):
        """
        Run an interaction iteration using the given action.
        """
        new_x, new_y = self.get_new_coordinates(action)
        self.time_step_id += 1
        if self.is_available(new_x, new_y):
            done = self.is_terminal_tile(new_x, new_y) or self.time_step_id > settings.max_time_steps
            reward = -1 if not done else 1
            self.agent_coordinates = new_x, new_y
            return self.get_state(self.agent_coordinates[0], self.agent_coordinates[1]), reward, done, None
        else:
            done = self.time_step_id > settings.max_time_steps
            return self.get_state(self.agent_coordinates[0], self.agent_coordinates[1]), -1, done, None

    def reset(self):
        """
        Reset the environment
        """
        self.agent_coordinates = self.start_coordinates
        self.time_step_id = 0
        return self.get_state(self.agent_coordinates[0], self.agent_coordinates[1])

    def get_oracle(self, coordinates: bool = False) -> list:
        """
        Return an oracle as a list of every possible states inside the environment.
        param coordinates: is a boolean that indicates if we want the result oracle to contain tuples of coordinates
            or states (if coordinates = False, default value).
        """
        oracle = []
        for y in range(self.height):
            for x in range(self.width):
                if self.is_available(x, y):
                    if not coordinates:
                        oracle.append(self.get_state(x, y))
                    else:
                        oracle.append((x, y))
        return oracle

    def get_available_positions(self, coordinates: tuple) -> list:
        """
        return an list of every available coordinates from the given one (used for A*).
        """
        x, y = coordinates  # Make sure coordinates is a tuple

        available_coordinates = []
        if x < (self.width - 1):
            new_coord = (x + 1, y)
            if self.is_available(x + 1, y):
                available_coordinates.append((new_coord, Direction.RIGHT.value))
        if x > 0:
            new_coord = (x - 1, y)
            if self.is_available(x - 1, y):
                available_coordinates.append((new_coord, Direction.LEFT.value))

        if y < (self.height - 1):
            new_coord = (x, y + 1)
            if self.is_available(x, y + 1):
                available_coordinates.append((new_coord, Direction.BOTTOM.value))
        if y > 0:
            new_coord = (x, y - 1)
            if self.is_available(x, y - 1):
                available_coordinates.append((new_coord, Direction.TOP.value))

        return available_coordinates

    ###################################
    # ENVIRONMENT RENDERING FUNCTIONS #
    ###################################
    def get_color(self, x, y, ignore_agent=False):
        """
        Return the color associated with the type of the tile at coordinates x, y. Used to render the environment.
        """
        agent_x, agent_y = self.agent_coordinates
        if (agent_x, agent_y) == (x, y) and not ignore_agent:
            return Colors.AGENT.value
        else:
            tile_type = self.get_tile_type(x, y)
            if tile_type == TileType.START:
                return Colors.START.value
            elif tile_type == TileType.WALL:
                return Colors.WALL.value
            elif tile_type == TileType.EMPTY:
                return Colors.EMPTY.value
            elif tile_type == TileType.TERMINAL:
                return Colors.TERMINAL.value
            else:
                raise AttributeError("Unknown tile type")

    def set_tile_color(self, image_array: np.ndarray, x, y, color, tile_size=settings.tile_size,
                       border_size=settings.border_size) -> np.ndarray:
        """
        Set a tile color with the given color in the given image as a numpy array of pixels
        :param image_array: The image where the tile should be set
        :param x: X coordinate of the tile to set
        :param y: Y coordinate of the tile to set
        :param color: new color of the tile : numpy array [Red, Green, Blue]
        :param tile_size: size of the tile in pixels
        :param border_size: size of the tile's border in pixels
        :return: The new image
        """
        tile_img = np.zeros(shape=(tile_size, tile_size, 3), dtype=np.uint8)

        if border_size > 0:
            tile_img[:, :, :] = Colors.TILE_BORDER.value
            tile_img[border_size:-border_size, border_size:-border_size, :] = color
        else:
            tile_img[:, :, :] = color

        y_min = y * tile_size
        y_max = (y + 1) * tile_size
        x_min = x * tile_size
        x_max = (x + 1) * tile_size

        image_array[y_min:y_max, x_min:x_max, :] = tile_img

        return image_array

    def get_environment_background(self, tile_size=settings.tile_size, ignore_agent=True) -> np.ndarray:
        """
        Give an image (as a numpy array of pixels) of the environment background.
        Used to generate an image of the environment in the render function.
        :return: environment background -> np.ndarray
        """
        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for y in range(self.height):
            for x in range(self.width):
                cell_color = self.get_color(x, y, ignore_agent=ignore_agent)

                img = self.set_tile_color(img, x, y, cell_color)

        return img

    def render(self, mode='human', show=True):
        """
        Render the whole-grid human view
        """
        if show:
            plt.cla()
            plt.ion()
        img = self.get_environment_background(ignore_agent=False)
        agent_x, agent_y = self.agent_coordinates
        image = self.set_tile_color(img, agent_x, agent_y, Colors.AGENT.value)
        if show:
            plt.imshow(image, interpolation='nearest')
            plt.show()
            plt.pause(0.00001)
        return image
