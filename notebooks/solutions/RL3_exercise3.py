### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

import gymnasium as gym
import gymnasium.envs.toy_text.frozen_lake as fl

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("FrozenLake-v1", render_mode="ansi")

# Policy definition and parameters
pi0 = fl.RIGHT * np.ones((env.observation_space.n), dtype=int)
gamma = 0.9
alpha = 0.001

Qtrue, residuals = policy_Qeval_iter(pi0, 1e-4, 10000)
print("Qtrue:\n", Qtrue)
print("number of iterations:", residuals.size)
plt.plot(residuals)
plt.figure()
plt.semilogy(residuals)

# TD(0) evaluation of Qpi
# parameters
gamma = 0.9
alpha = 0.001
max_steps = 2000000
Q = np.transpose(np.tile(V, (4, 1)))

error = np.zeros(max_steps)
s, _ = env.reset()
for t in range(max_steps):
    a = np.random.randint(4)
    s2, r, d, _, _ = env.step(a)
    Q[s][a] = Q[s][a] + alpha * (r + gamma * Q[s2][fl.RIGHT] - Q[s][a])
    error[t] = np.max(np.abs(Q - Qtrue))
    if d:
        s, _ = env.reset()
    else:
        s = s2

print("Max error:", np.max(np.abs(Q - Qtrue)))
plt.figure()
plt.plot(error)
plt.figure()
plt.semilogy(error)
