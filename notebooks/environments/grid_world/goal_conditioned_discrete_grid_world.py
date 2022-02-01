import numpy as np
from matplotlib import pyplot as plt

from environments.grid_world import settings
from environments.grid_world.discrete_grid_world import DiscreteGridWorld
from environments.grid_world.utils.indexes import Colors
import random


class GoalConditionedDiscreteGridWorld(DiscreteGridWorld):
    def __init__(self, map_id=settings.map_id):
        super().__init__(map_id)
        self.goal_coordinates = None  # Agent coordinates inside the gris as tuple of integers (X, Y)
        self.goal = None  # Agent goal as a state (coordinates between 0.0 and 1.0)
        self.reset_goal()

    def reset_goal(self) -> np.ndarray:
        """
        Choose a goal for the agent.
        :return: the goal
        """
        oracle = self.get_oracle(coordinates=True)  # Free of unreachable states
        self.goal_coordinates = random.choice(oracle)
        self.goal = self.get_state(*self.goal_coordinates)
        return self.goal

    def goal_reached(self) -> bool:
        """
        Return a boolean True if the agent state is on the goal (and exactly on the goal since our state space is
        discrete here in reality), and false otherwise.
        """
        return self.agent_coordinates == self.goal_coordinates

    def step(self, action) -> (np.ndarray, float, bool, object):
        new_x, new_y = self.get_new_coordinates(action)
        self.time_step_id += 1
        if self.is_available(new_x, new_y):
            self.agent_coordinates = new_x, new_y
            reached = self.goal_reached()
            reward = 0.if not reached else 1.
            done = reached or self.time_step_id > settings.max_time_steps
            return self.get_state(self.agent_coordinates[0], self.agent_coordinates[1]), reward, done, None
        else:
            done = self.time_step_id > settings.max_time_steps
            return self.get_state(self.agent_coordinates[0], self.agent_coordinates[1]), 0., done, None

    def reset(self) -> (np.ndarray, np.ndarray):
        """
        Return the initial state, and the selected goal.
        """
        self.reset_goal()
        self.time_step_id = 0
        return super().reset(), self.goal

    def render(self, mode='human', show=True) -> np.ndarray:
        """
        Render the whole-grid human view (get view from super class then add the goal over the image)
        """
        if show:
            plt.cla()
            plt.ion()
        img = super().render(mode=mode, show=False)
        goal_x, goal_y = self.goal_coordinates
        image = self.set_tile_color(img, goal_x, goal_y, Colors.GOAL.value)

        if show:
            plt.imshow(image, interpolation='nearest')
            plt.show()
            plt.pause(0.00001)
        return image

