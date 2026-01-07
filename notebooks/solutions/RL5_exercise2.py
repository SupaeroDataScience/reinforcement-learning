### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = cartpole.observation_space.shape[0]
n_action = cartpole.action_space.n
nb_neurons = 24

DQN = torch.nn.Sequential(
    nn.Linear(state_dim, nb_neurons),
    nn.ReLU(),
    nn.Linear(nb_neurons, nb_neurons),
    nn.ReLU(),
    nn.Linear(nb_neurons, n_action),
).to(device)
