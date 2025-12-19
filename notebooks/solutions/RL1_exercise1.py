### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

import gymnasium as gym
import gymnasium.envs.toy_text.frozen_lake as fl
import numpy as np
from tqdm import trange

env = gym.make("FrozenLake-v1", render_mode="ansi")

nb_episodes = 100000
horizon = 200
gamma = 0.9
Vepisode = np.zeros(nb_episodes)
for i in trange(nb_episodes):
    env.reset()
    for t in range(horizon):
        next_state, r, done, _, _ = env.step(fl.RIGHT)
        Vepisode[i] += gamma**t * r
        if done:
            break

print("Value estimate:", np.mean(Vepisode))
print("Value variance:", np.std(Vepisode))
