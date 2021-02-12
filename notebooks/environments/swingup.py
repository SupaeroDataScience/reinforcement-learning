import gym
from gym import logger
import numpy as np

class CartPoleSwingUp(gym.Wrapper):
    def __init__(self, env=gym.make('CartPole-v1'), **kwargs):
        super(CartPoleSwingUp, self).__init__(env, **kwargs)
        self.theta_dot_threshold = 4*np.pi

    def reset(self):
        self.env.env.state = [0, 0, np.pi, 0] + super().reset()
        return np.array(self.env.env.state)

    def step(self, action):
        state, reward, done, _ = super().step(action)
        self.env.env.steps_beyond_done = None
        x, x_dot, theta, theta_dot = state
        theta = (theta+np.pi)%(2*np.pi)-np.pi
        self.env.env.state = [x, x_dot, theta, theta_dot]
        
        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta_dot < -self.theta_dot_threshold \
               or theta_dot > self.theta_dot_threshold
        
        if done:
            # game over
            reward = -10.
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
            elif self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
                self.steps_beyond_done += 1
        else:
            if -self.theta_threshold_radians < theta and theta < self.theta_threshold_radians:
                # pole upright
                reward = 1.
            else:
                # pole swinging
                reward = 0.

        return np.array([x, x_dot, theta, theta_dot]), reward, done, {}
