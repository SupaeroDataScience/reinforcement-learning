### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

import gymnasium as gym
import gymnasium.envs.toy_text.frozen_lake as fl

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

env = gym.make("FrozenLake-v1", render_mode="ansi")

# parameters
gamma = 0.9
alpha = 0.001
max_steps = 1000
max_episodes = 100000
V = np.zeros(env.observation_space.n)

# error plotting
error = np.zeros(max_episodes)  # used to track the convergence to V_pi0...
cumulated_steps = np.zeros(max_episodes)  # ... against the number of samples

for ep in trange(max_episodes):
    s, _ = env.reset()
    episode = []
    # Run episode
    for t in range(max_steps):
        s2, r, d, _, _ = env.step(fl.RIGHT)
        episode.append([s, r])
        if d:
            cumulated_steps[ep] = cumulated_steps[ep - 1] + t
            break
        else:
            s = s2
    # Update values
    T = len(episode)
    G = np.zeros(T)
    G[-1] = episode[-1][1]
    s = episode[-1][0]
    V[s] = V[s] + alpha * (G[-1] - V[s])
    for t in range(-2, -T - 1, -1):
        G[t] = episode[t][1] + gamma * G[t + 1]
        s = episode[t][0]
        V[s] = V[s] + alpha * (G[t] - V[s])
    error[ep] = np.max(np.abs(V - V_pi0))

print(V)
print(V_pi0)
plt.plot(cumulated_steps, error)
plt.figure()
plt.semilogy(cumulated_steps, error)
