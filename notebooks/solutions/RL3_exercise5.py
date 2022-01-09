### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

def TD_Qeval(pi, max_steps, env, alpha, gamma, Q=None, Qtrue=None):
    error = np.zeros((max_steps))
    if (Q is None):
        Q = np.zeros((env.observation_space.n, env.action_space.n))
    x = env.reset()
    for t in range(max_steps):
        a = np.random.randint(4)
        y,r,d,_ = env.step(a)
        Q[x][a] = Q[x][a] + alpha * (r+gamma*Q[y][pi[y]]-Q[x][a])
        if(Qtrue is not None):
            error[t] = np.max(np.abs(Q-Qtrue))
        if d==True:
            x = env.reset()
        else:
            x=y
    return Q, error