### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

import gym
import gym.envs.toy_text.frozen_lake as fl
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
gamma = 0.9

def policy_eval_iter_mat2(pi, epsilon, max_iter):
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
    W = np.zeros((env.observation_space.n))
    residuals = np.zeros((max_iter))
    for i in range(max_iter):
        W = r_pi + gamma * np.dot(P_pi, V)
        residuals[i] = np.max(np.abs(W-V))
        V[:] = W
        if residuals[i]<epsilon:
            residuals = residuals[:i+1]
            break
    return V, residuals

V_pi0, residuals = policy_eval_iter_mat2(pi0,1e-4,10000)
print(V_pi0)
plt.plot(residuals)
plt.figure()
plt.semilogy(residuals)
print("number of iterations:", residuals.size)
print("last residual", residuals[-1])
