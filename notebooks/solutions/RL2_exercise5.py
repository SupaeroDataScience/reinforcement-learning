### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

import gymnasium as gym
import gymnasium.envs.toy_text.frozen_lake as fl
import numpy as np

env = gym.make("FrozenLake-v1", render_mode="ansi")
gamma = 0.9
pi0 = fl.RIGHT * np.ones(env.observation_space.n)

def policy_eval_iter(pi, max_iter):
    """Compute V(pi) by applying the Bellman evaluation operator max_iter times"""
    V = np.zeros(env.observation_space.n)
    W = np.zeros(env.observation_space.n)
    for i in range(max_iter):
        # Compute W = P * (r + gamma V)
        for s in range(env.observation_space.n):
            W[s] = 0
            outcomes = env.unwrapped.P[s][pi[s]]
            # W[x] = sum_s2 P[s2] (r_s2 + gamma V[s2])
            for o in outcomes:
                p  = o[0]
                s2 = o[1]
                r  = o[2]
                W[s] += p * (r + gamma * V[s2])
        V[:] = W
    return V

def policy_eval_iter_mat(pi, max_iter):
    """Compute V(pi) by applying the Bellman evaluation operator max_iter times in matrix form"""
    # Build r and P
    r_pi = np.zeros(env.observation_space.n)
    P_pi = np.zeros((env.observation_space.n, env.observation_space.n))
    for s in range(env.observation_space.n):
        outcomes = env.unwrapped.P[s][pi[s]]
        for o in outcomes:
            p  = o[0]
            s2 = o[1]
            r  = o[2]
            P_pi[s, s2] += p
            r_pi[s] += r * p
    # Compute V
    V = np.zeros(env.observation_space.n)
    for i in range(max_iter):
        V = r_pi + gamma * np.dot(P_pi, V)
    return V

V_pi0 = policy_eval_iter(pi0, 10000)
print(V_pi0)

V_pi0 = policy_eval_iter_mat(pi0, 10000)
print(V_pi0)
