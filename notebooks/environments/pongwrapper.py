import ale_py
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing

gym.register_envs(ale_py)

class PongWrapper(AtariPreprocessing):
    def __init__(self, **kwargs):
        render_mode = kwargs.pop("render_mode", None)
        env = gym.make("ALE/Pong-v5", frameskip=1, render_mode=render_mode)
        kwargs["screen_size"] = (84, 110)
        super().__init__(env, **kwargs)

    def step(self, action):
        return super().step(4 + action)

    def _get_obs(self):
        return super()._get_obs()[17:101,:]
        if self.frame_skip > 1:  # more efficient in-place pooling
            np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])
        obs = cv2.resize(self.obs_buffer[0], (84, 110), interpolation=cv2.INTER_AREA)[17:101,:]

        if self.scale_obs:
            obs = np.asarray(obs, dtype=np.float32) / 255.0
        else:
            obs = np.asarray(obs, dtype=np.uint8)
        return obs
