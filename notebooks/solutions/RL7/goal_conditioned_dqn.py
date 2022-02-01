import copy

import gym
import numpy as np
import torch
from torch import nn, optim
from torch.nn import ReLU

from agents.nn.multilayers_perceptron import MLP


class GCDQNAgent(GoalConditionedAgent):
    """
    An agent that learn an approximated Q-Function using a neural network.
    This Q-Function is used to find the best action to execute in a given state.
    """

    def __init__(self, state_space, action_space, goal_size=None, name="DQN",
                 gamma=0.95, epsilon_min=0.01, epsilon_max=1., epsilon_decay_period=1000, epsilon_decay_delay=20,
                 buffer_size=1000000, learning_rate=0.001, update_target_freq=100, batch_size=125,
                 layer_1_size=250, layer_2_size=200, nb_gradient_steps=1):

        assert isinstance(action_space, gym.spaces.Discrete)  # Make sure our action space is discrete
        super().__init__(state_space, action_space, goal_size=goal_size, name=name)

        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_decay_delay = epsilon_decay_delay
        self.epsilon = None

        #  NEW, goals will be stored inside the replay buffer. We need a specific one with enough place to do so
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.update_target_freq = update_target_freq
        self.nb_gradient_steps = nb_gradient_steps

        self.epsilon_step = (epsilon_max - self.epsilon_min) / epsilon_decay_period
        self.total_steps = 0

        # NEW, The input state size is multiplied by two because we need to also take the goal as input
        self.model = MLP(self.state_size + self.goal_size, layer_1_size, ReLU(), layer_2_size, ReLU(), 
                         self.nb_actions, learning_rate=learning_rate, optimizer_class=optim.Adam, 
                         device=self.device).float()

        self.criterion = torch.nn.SmoothL1Loss()
        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.update_target_freq = update_target_freq

    def on_simulation_start(self):
        self.epsilon = self.epsilon_max

    def action(self, state):
        if self.time_step_id > self.epsilon_decay_delay:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_step)

        if np.random.rand() < self.epsilon:  # Epsilon greedy
            action = np.random.randint(self.nb_actions)
        else:
            # greedy_action(self.model, state) function in RL5 notebook
            with torch.no_grad():
                goal_conditioned_state = np.concatenate((state, self.current_goal), axis=-1)
                Q = self.model(goal_conditioned_state)
                action = torch.argmax(Q).item()
        return action

    def on_action_stop(self, action, new_state, reward, done):
        # NEW store the current goal, so it can be associated with samples
        self.replay_buffer.append(self.last_state, action, reward, new_state, done, self.current_goal)
        self.learn()
        super().on_action_stop(action, new_state, reward, done)  # Replace self.last_state by the new_state

    def learn(self):
        for _ in range(self.nb_gradient_steps):
            # gradient_step() function in RL5 notebook
            if len(self.replay_buffer) > self.batch_size:
                #  NEW, samples from buffer contains goals
                states, actions, rewards, new_states, dones, goals = self.replay_buffer.sample(self.batch_size)

                # NEW concatenate states and goals, because we need to put them inside our model
                goal_conditioned_states = torch.concat((states, goals), dim=-1)
                goal_conditioned_new_states = torch.concat((new_states, goals), dim=-1)

                q_prime = self.target_model(goal_conditioned_new_states).max(1)[0].detach()
                update = rewards + self.gamma * (1 - dones) * q_prime
                q_s_a = self.model(goal_conditioned_states).gather(1, actions.to(torch.long).unsqueeze(1))
                loss = self.criterion(q_s_a, update.unsqueeze(1))
                self.model.learn(loss)

        # update target network if needed
        if self.time_step_id % self.update_target_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

