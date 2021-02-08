### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

import gym
import gym.envs.toy_text.frozen_lake as fl
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
gamma = 0.9
m = 500

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

def greedyQpolicy(Q):
    pi = np.zeros((env.observation_space.n),dtype=np.int)
    for s in range(env.observation_space.n):
        pi[s] = np.argmax(Q[s,:])
    return pi

def modified_policy_iteration(pi0,m,max_iter):
    policies = np.zeros((max_iter, env.observation_space.n))
    policies[0,:] = np.copy(pi0)
    for i in range(max_iter-1):
        Vpi = policy_eval_iter_mat(policies[i],m)
        Qpi = Q_from_V(Vpi)
        policies[i+1,:] = greedyQpolicy(Qpi)
        if np.array_equal(policies[i,:],policies[i+1,:]):
            policies = policies[:i,:]
            break
    return policies

def print_policy(pi):
    for row in range(env.unwrapped.nrow):
        for col in range(env.unwrapped.ncol):
            print(actions[pi[to_s(row,col)]], end='')
        print()
    return

pi0 = fl.RIGHT*np.ones((env.observation_space.n))
print_policy(pi0)
policies = modified_policy_iteration(pi0,m,10)
print("number of iterations:", policies.shape[0])
print_policy(policies[-1,:])
_=env.render()