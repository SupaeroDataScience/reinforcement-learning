### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

import gym
import gym.envs.toy_text.frozen_lake as fl
import numpy as np

env = gym.make('FrozenLake-v0')
gamma = 0.9
pi0 = fl.RIGHT*np.ones((env.observation_space.n))

def policy_eval_iter(pi, max_iter):
    V = np.zeros((env.observation_space.n))
    W = np.zeros((env.observation_space.n))
    for i in range(max_iter):
        for x in range(env.observation_space.n):
            W[x]=0
            outcomes = env.unwrapped.P[x][pi[x]]
            # W[x] = sum_y P[y] (r_y + gamma V[y])
            for o in outcomes:
                p = o[0]
                y = o[1]
                r = o[2]
                W[x] += p * (r+gamma*V[y])
        V[:] = W
    return V

def policy_eval_iter_mat(pi, max_iter):
    # build r and P
    r_pi = np.zeros((env.observation_space.n))
    P_pi = np.zeros((env.observation_space.n, env.observation_space.n))
    for x in range(env.observation_space.n):
        outcomes = env.unwrapped.P[x][pi[x]]
        for o in outcomes:
            p = o[0]
            y = o[1]
            r = o[2]
            P_pi[x,y] += p
            r_pi[x] += r*p
    # Compute V
    V = np.zeros((env.observation_space.n))
    for i in range(max_iter):
        V = r_pi + gamma * np.dot(P_pi, V)
    return V

V_pi0 = policy_eval_iter(pi0,10000)
print(V_pi0)

V_pi0 = policy_eval_iter_mat(pi0,10000)
print(V_pi0)
