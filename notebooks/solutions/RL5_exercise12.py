### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

# Let's reset the Q function

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange


class AtariCNN(torch.nn.Module):
    def __init__(self, in_channels=4, n_actions=2):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = torch.nn.Linear(7 * 7 * 64, 512)
        self.head = torch.nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)


config = {
    "observation_space": pong.observation_space.shape[0],
    "nb_actions": pong.action_space.n,
    "learning_rate": 0.001,
    "gamma": 0.95,
    "buffer_size": 1000000,
    "epsilon_min": 0.1,
    "epsilon_max": 1.0,
    "epsilon_stop": 1000000,
    "epsilon_decay_period": 10000,
    "epsilon_delay_decay": 0,
    "batch_size": 32,
    "gradient_steps": 10,
    "update_target_freq": 10000,
    "nb_trials": 0,
}

AtariDQN = AtariCNN()

agent = DQNAgent(config, AtariDQN)

# pre-fill the replay buffer
x, _ = pong.reset()
for _ in trange(50000):
    a = np.random.randint(2)
    y, r, d, _, _ = pong.step(a)
    agent.memory.append(x, a, r, y, d)
    if d:
        x, _ = pong.reset()
    else:
        x = y

# train
ep_length, disc_rewards, tot_rewards, Q0 = agent.train(pong, 30)
