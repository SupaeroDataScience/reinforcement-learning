### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

# Replay buffer class
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.data, batch_size)

    def __len__(self):
        return len(self.data)

# Testing insertion in the ReplayBuffer class
from tqdm import trange
replay_buffer_size = int(1e6)
nb_samples = int(2e6)

memory = ReplayBuffer(replay_buffer_size)
state = cartpole.reset()
for _ in trange(nb_samples):
    action = cartpole.action_space.sample()
    next_state, reward, done, _ = cartpole.step(action)
    memory.append(state, action, reward, next_state, done)
    if done:
        state = cartpole.reset()
    else:
        state = next_state

print(len(memory))

# Testing sampling in the ReplayBuffer class
nb_batches = int(1e4)
batch_size = 50
import random

for _ in trange(nb_batches):
    batch = memory.sample(batch_size)

print(memory.sample(2))