### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

import gymnasium as gym
import gymnasium.envs.toy_text.frozen_lake as fl

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("FrozenLake-v1", render_mode="ansi")

# parameters
gamma = 0.9
alpha = 0.001
max_steps = 2000000
V = np.zeros(env.observation_space.n)

# error plotting
error = np.zeros(max_steps)  # used to track the convergence to V_pi0

s, _ = env.reset()
for t in trange(max_steps):
    s2, r, d, _, _ = env.step(fl.RIGHT)
    V[s] = V[s] + alpha * (r + gamma * V[s2] - V[s])
    error[t] = np.max(np.abs(V - V_pi0))
    if d:
        s, _ = env.reset()
    else:
        s = s2

print(V)
print(V_pi0)
plt.plot(error)
plt.figure()
plt.semilogy(error)
