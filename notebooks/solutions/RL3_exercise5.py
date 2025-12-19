### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

import numpy as np
from tqdm import trange

def TD_Qeval(pi, max_steps, env, alpha, gamma, Q=None, Qtrue=None):
    error = np.zeros(max_steps)
    if Q is None:
        Q = np.zeros((env.observation_space.n, env.action_space.n))
    s = env.reset()
    for t in trange(max_steps):
        a = np.random.randint(4)
        s2, r, d, _, _ = env.step(a)
        Q[s][a] = Q[s][a] + alpha * (r + gamma * Q[s2][pi[s2]] - Q[s][a])
        if Qtrue is not None:
            error[t] = np.max(np.abs(Q - Qtrue))
        if d:
            s = env.reset()
        else:
            s = s2
    return Q, error
