### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

import gym
import gym.envs.toy_text.frozen_lake as fl
import numpy as np

env = gym.make('FrozenLake-v0')

nb_episodes = 100000
horizon = 200
gamma = 0.9
Vepisode = np.zeros(nb_episodes)
for i in range(nb_episodes):
    env.reset()
    for t in range(horizon):
        next_state, r, done,_ = env.step(fl.RIGHT)
        Vepisode[i] += gamma**t * r
        if done:
            break
print("value estimate:", np.mean(Vepisode))
print("value variance:", np.std(Vepisode))
