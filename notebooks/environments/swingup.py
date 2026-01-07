import warnings
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import logger, spaces
from gymnasium.core import WrapperActType, WrapperObsType


class CartPoleSwingUp(gym.Wrapper):
    def __init__(self, **kwargs):
        env = gym.make("CartPole-v1", **kwargs)
        super().__init__(env)

        self.theta_dot_threshold = 4 * np.pi
        high = np.array(
            [
                self.unwrapped.x_threshold * 2,
                np.inf,
                np.inf,
                self.theta_dot_threshold * 2,
            ],
            dtype=np.float32,
        )
        self.unwrapped.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        render_mode_bak = self.unwrapped.render_mode
        self.unwrapped.render_mode = "none"
        state, info = super().reset(seed=seed, options=options)
        self.unwrapped.render_mode = render_mode_bak

        state[2] += np.pi
        self.unwrapped.state = state

        self.steps_beyond_done = 0

        if self.render_mode == "human":
            self.render()
        return np.array(self.unwrapped.state), info

    def step(
        self, action: WrapperActType,
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning, 214)
            state, _, _, trunc, info = super().step(action)

        x, x_dot, theta, theta_dot = state
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        self.unwrapped.state = [x, x_dot, theta, theta_dot]

        terminated = (
            x < -self.unwrapped.x_threshold
            or x > self.unwrapped.x_threshold
            or theta_dot < -self.theta_dot_threshold
            or theta_dot > self.theta_dot_threshold
        )

        if terminated:
            reward = -10.
            if self.steps_beyond_done == 1:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
        else:
            if -self.unwrapped.theta_threshold_radians < theta < self.unwrapped.theta_threshold_radians:
                # pole upright
                reward = 1.
            else:
                # pole swinging
                reward = 0.

        return np.array([x, x_dot, theta, theta_dot]), reward, terminated, trunc, info
