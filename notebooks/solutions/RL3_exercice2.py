### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

import gym
import gym.envs.toy_text.frozen_lake as fl
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

# parameters
gamma = 0.9
alpha = 0.001
max_steps = 2000000
V = np.zeros((env.observation_space.n))

# error plotting
error = np.zeros((max_steps)) # used to track the convergence to V_pi0

x = env.reset()
for t in range(max_steps):
    y,r,d,_ = env.step(fl.RIGHT)
    V[x] = V[x] + alpha * (r+gamma*V[y]-V[x])
    error[t] = np.max(np.abs(V-V_pi0))
    if d==True:
        x = env.reset()
    else:
        x=y

print(V)
print(V_pi0)
plt.plot(error)
plt.figure()
plt.semilogy(error);