### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

# Let's reset the Q function

import torch
import torch.nn as nn
from environments.swingup import CartPoleSwingUp

swingup = CartPoleSwingUp()

config = {'observation_space': swingup.observation_space.shape[0],
          'nb_actions': swingup.action_space.n,
          'learning_rate': 0.001,
          'gamma': 0.95,
          'buffer_size': 1000000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_stop': 10000,
          'epsilon_delay_decay': 100,
          'batch_size': 20,
          'gradient_steps': 10,
          'update_target_freq': 100,
          'nb_trials': 0}

DQN = torch.nn.Sequential(nn.Linear(swingup.observation_space.shape[0], 24),
                          nn.ReLU(),
                          nn.Linear(24, 24),
                          nn.ReLU(), 
                          nn.Linear(24, swingup.action_space.n)).to(device)

agent = DQN_agent(config, DQN)
ep_length, disc_rewards, tot_rewards, Q0 = agent.train(swingup, 30)
torch.save(DQN.state_dict(), "swingup_dqn.pth")

DQN.load_state_dict(torch.load("swingup_dqn.pth"))
x = swingup.reset()
swingup.render()
tot_rew = 0
for i in range(1000):
    a = greedy_action(DQN, x)
    y, r, d, _ = swingup.step(a)
    swingup.render()
    x=y
    tot_rew += r
    if d:
        break

print(i)
print(tot_rew)

swingup.close()