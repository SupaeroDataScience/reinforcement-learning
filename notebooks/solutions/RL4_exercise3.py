### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

import gymnasium as gym
import gymnasium.envs.toy_text.frozen_lake as fl
import numpy as np

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

# Policy definition and parameters
Q0  = np.zeros((env.observation_space.n,env.action_space.n))
gamma = 0.9
alpha = 0.001
max_steps=2000000
update_period=1

Q = np.copy(Q0)
x = env.reset()
for t in range(max_steps):
    if t%update_period==0:
        pi = greedyQpolicy(Q)
    a = np.random.randint(4)
    y,r,d,_,_ = env.step(a)
    Q[x][a] = Q[x][a] + alpha * (r+gamma*Q[y][pi[y]]-Q[x][a])
    if d==True:
        x = env.reset()
    else:
        x=y

pi = greedyQpolicy(Q)
print_policy(pi)
Qpi = Q_from_V(policy_eval_iter_mat(pi,1000))
print(np.max(np.abs(Qpi-Qstar)))
print(pi-pi_star)
