### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

import gymnasium as gym
import gymnasium.envs.toy_text.frozen_lake as fl
import numpy as np

env = gym.make("FrozenLake-v1")
gamma = 0.9

def policy_eval_lin(pi):
    # Build r and P
    r_pi = np.zeros(env.observation_space.n)
    P_pi = np.zeros((env.observation_space.n, env.observation_space.n))
    for s in range(env.observation_space.n):
        outcomes = env.unwrapped.P[s][pi[s]]
        for o in outcomes:
            p = o[0]
            y = o[1]
            r = o[2]
            P_pi[s, y] += p
            r_pi[s] += r * p
    # Compute V
    I = np.eye(env.observation_space.n)
    return np.dot(np.linalg.inv(I - gamma * P_pi), r_pi)

pi0 = fl.RIGHT * np.ones(env.observation_space.n)

V_pi0 = policy_eval_lin(pi0)
print(V_pi0)
