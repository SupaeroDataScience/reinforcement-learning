### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

import numpy as np
from tqdm import trange

# Let's restart from Qpi0
Qsarsa = Qpi0
max_steps = 5000000

def epsilon_greedy(Q, s, epsilon):
    if np.random.rand() <= epsilon:  # random action
        return np.random.randint(env.action_space.n - 1)
    return np.argmax(Q[s, :])

def policy_Qeval_iter(pi, epsilon, max_iter):
    Q1 = np.zeros((env.observation_space.n, env.action_space.n))
    Q2 = np.zeros((env.observation_space.n, env.action_space.n))
    residuals = np.zeros(max_iter)
    for i in trange(max_iter):
        for x in range(env.observation_space.n):
            for a in range(env.action_space.n):
                Q2[x][a] = 0
                outcomes = env.unwrapped.P[x][a]
                for o in outcomes:
                    p = o[0]
                    y = o[1]
                    r = o[2]
                    Q2[x][a] += p * (r + gamma * Q1[y][pi[y]])
        residuals[i] = np.max(np.abs(Q2 - Q1))
        Q1[:] = Q2
        if residuals[i] < epsilon:
            residuals = residuals[: i + 1]
            break
    return Q1, residuals


# SARSA
count = np.zeros((env.observation_space.n, env.action_space.n))  # to track update frequencies
epsilon = 1
s, _ = env.reset()
a = epsilon_greedy(Qsarsa, s, epsilon)
for t in trange(max_steps):
    if (t + 1) % 1000000 == 0:
        epsilon = epsilon / 2
    s2, r, d, _, _ = env.step(a)
    aa = epsilon_greedy(Qsarsa, s2, epsilon)
    Qsarsa[s][a] = Qsarsa[s][a] + alpha * (r + gamma * Qsarsa[s2][aa] - Qsarsa[s][a])
    count[s][a] += 1
    if d:
        s, _ = env.reset()
        a = epsilon_greedy(Qsarsa, s, epsilon)
    else:
        s = s2
        a = aa

# SARSA's final value function and policy
print("Max error:", np.max(np.abs(Qsarsa - Qstar)))

print("Final epsilon:", epsilon)
pi_sarsa = greedyQpolicy(Qsarsa)
print("Greedy SARSA policy:")
print_policy(pi_sarsa)
print("Difference between pi_sarsa and pi_star (recall that there are several optimal policies):")
print(pi_sarsa - pi_star)
Qpi_sarsa, residuals = policy_Qeval_iter(pi_sarsa, 1e-4, 10000)
print("Max difference in value between pi_sarsa and pi_star:", np.max(np.abs(Qpi_sarsa - Qstar)))
print("Min difference in value between pi_sarsa and pi_star:", np.min(np.abs(Qpi_sarsa - Qstar)))
