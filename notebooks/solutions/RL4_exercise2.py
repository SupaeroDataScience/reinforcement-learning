### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

import gymnasium as gym
import gymnasium.envs.toy_text.frozen_lake as fl
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1')

def greedyQpolicy(Q):
    pi = np.zeros(env.observation_space.n,dtype=int)
    for s in range(env.observation_space.n):
        pi[s] = np.argmax(Q[s,:])
    return pi

def to_s(row,col):
    return row*env.unwrapped.ncol+col

def print_policy(pi):
    actions = {fl.LEFT: '\u2190', fl.DOWN: '\u2193', fl.RIGHT: '\u2192', fl.UP: '\u2191'}
    for row in range(env.unwrapped.nrow):
        for col in range(env.unwrapped.ncol):
            print(actions[pi[to_s(row,col)]], end='')
        print()
    return

def value_iteration(V,epsilon,max_iter):
    W = np.copy(V)
    residuals = np.zeros(max_iter)
    for i in range(max_iter):
        for s in range(env.observation_space.n):
            Q = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                outcomes = env.unwrapped.P[s][a]
                for o in outcomes:
                    p  = o[0]
                    s2 = o[1]
                    r  = o[2]
                    Q[a] += p*(r+gamma*V[s2])
            W[s] = np.max(Q)
            #print(W[s])
        residuals[i] = np.max(np.abs(W-V))
        #print("abs", np.abs(W-V))
        np.copyto(V,W)
        if residuals[i]<epsilon:
            residuals = residuals[:i+1]
            break
    return V, residuals

def Q_from_V(V):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            outcomes = env.unwrapped.P[s][a]
            for o in outcomes:
                p  = o[0]
                s2 = o[1]
                r  = o[2]
                Q[s,a] += p*(r+gamma*V[s2])
    return Q

def policy_eval_iter_mat(pi, max_iter):
    # build r and P
    r_pi = np.zeros(env.observation_space.n)
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
    V = np.zeros(env.observation_space.n)
    for i in range(max_iter):
        V = r_pi + gamma * np.dot(P_pi, V)
    return V

# Policy definition and parameters
Q0  = np.zeros((env.observation_space.n,env.action_space.n))
gamma = 0.9
alpha = 0.001
max_steps=1000000
update_period=1
eval_period=1000
optimality_gap = []

# Model-based optimisation
Vinit = np.zeros(env.observation_space.n)
Vstar,residuals = value_iteration(Vinit,1e-4,1000)
Qstar = Q_from_V(Vstar)

Q = np.copy(Q0)
x = env.reset()
for t in range(max_steps):
    # update policy every N steps
    if t%update_period==0:
        pi = greedyQpolicy(Q)
    # evaluate policy (just for monitoring)
    if t%eval_period==0:
        Qpi = Q_from_V(policy_eval_iter_mat(pi, 1000))
        optimality_gap.append(np.max(np.abs(Qpi-Qstar)))
    # random behavior policy
    a = np.random.randint(4)
    y,r,d,_,_ = env.step(a)
    # TD(0) update
    Q[x][a] = Q[x][a] + alpha * (r+gamma*Q[y][pi[y]]-Q[x][a])
    if d==True:
        x = env.reset()
    else:
        x=y

pi = greedyQpolicy(Q)
Qpi = Q_from_V(policy_eval_iter_mat(pi, 1000))
optimality_gap.append(np.max(np.abs(Qpi-Qstar)))
print_policy(pi)
print(pi-greedyQpolicy(Qstar)) # remember there may be several optimal policies
plt.plot(optimality_gap);
