import gym
from gym.wrappers import AtariPreprocessing
import cv2
import numpy as np

class PongWrapper(AtariPreprocessing):
    def __init__(self, env=gym.make('PongNoFrameskip-v4'), **kwargs):
        super(PongWrapper, self).__init__(env, **kwargs)

    def step(self, action):
        return super(PongWrapper, self).step(4 + action)

    def _get_obs(self):
        if self.frame_skip > 1:  # more efficient in-place pooling
            np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])
        obs = cv2.resize(self.obs_buffer[0], (84, 110), interpolation=cv2.INTER_AREA)[17:101,:]

        if self.scale_obs:
            obs = np.asarray(obs, dtype=np.float32) / 255.0
        else:
            obs = np.asarray(obs, dtype=np.uint8)
        return obs
