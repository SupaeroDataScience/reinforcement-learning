### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

import gymnasium as gym
import gymnasium.envs.toy_text.frozen_lake as fl
import numpy as np

env = gym.make("FrozenLake-v1")

def greedyQpolicy(Q):
    pi = np.zeros(env.observation_space.n, dtype=int)
    for s in range(env.observation_space.n):
        pi[s] = np.argmax(Q[s, :])
    return pi

pi = greedyQpolicy(Q)
print(pi)
print_policy(pi)
