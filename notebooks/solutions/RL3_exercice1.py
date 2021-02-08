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
max_steps = 1000
max_episodes = 100000
V = np.zeros((env.observation_space.n))

# error plotting
error = np.zeros((max_episodes)) # used to track the convergence to V_pi0...
cumulated_steps = np.zeros((max_episodes)) # ... against the number of samples

for ep in range(max_episodes):
    x = env.reset()
    episode = []
    # Run episode
    for t in range(max_steps):
        y,r,d,_ = env.step(fl.RIGHT)
        episode.append([x,r])
        if d==True:
            cumulated_steps[ep] = cumulated_steps[ep-1] + t
            break
        else:
            x=y
    # Update values
    T = len(episode)
    G = np.zeros((T))
    G[-1] = episode[-1][1]
    x = episode[-1][0]
    V[x] = V[x] + alpha * (G[-1] - V[x])
    for t in range(-2,-T-1,-1):
        G[t] = episode[t][1] + gamma*G[t+1]
        x = episode[t][0]
        V[x] = V[x] + alpha * (G[t] - V[x])
    error[ep] = np.max(np.abs(V-V_pi0))

print(V)
print(V_pi0)
plt.plot(cumulated_steps,error)
plt.figure()
plt.semilogy(cumulated_steps,error);
