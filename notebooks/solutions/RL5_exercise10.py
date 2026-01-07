### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

# Let's reset the Q function

import torch
import torch.nn as nn
import gymnasium as gym

cartpole = gym.make("CartPole-v1")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = cartpole.observation_space.shape[0]
n_action = cartpole.action_space.n
nb_neurons = 24

# reset the Q function
DQN = torch.nn.Sequential(
    nn.Linear(state_dim, nb_neurons),
    nn.ReLU(),
    nn.Linear(nb_neurons, nb_neurons),
    nn.ReLU(),
    nn.Linear(nb_neurons, n_action),
).to(device)

config = {
    "observation_space": cartpole.observation_space.shape[0],
    "nb_actions": cartpole.action_space.n,
    "learning_rate": 0.001,
    "gamma": 0.95,
    "buffer_size": 1000000,
    "epsilon_min": 0.01,
    "epsilon_max": 1.0,
    "epsilon_decay_period": 1000,
    "epsilon_delay_decay": 20,
    "batch_size": 20,
    "gradient_steps": 10,
    "update_target_freq": 100,
    "nb_trials": 50,
}

agent = DQNAgent(config, DQN)
ep_length, disc_rewards, tot_rewards, Q0 = agent.train(cartpole, 100)

plt.plot(ep_length, label="Training episode length")
plt.plot(tot_rewards, label="MC eval of total reward")
plt.legend()
plt.figure()
plt.plot(disc_rewards, label="MC eval of discounted reward")
plt.plot(Q0, label="average $max_a Q(s_0)$")
plt.legend()
