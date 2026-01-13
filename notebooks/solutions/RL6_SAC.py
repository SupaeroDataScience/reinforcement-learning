# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch


class SACAgent:
    def __init__(
        self,
        state_space,
        action_space,
        actor_lr=0.001,
        critic_lr=0.002,
        gamma=0.99,
        buffer_size=100000,
        tau=0.01,
        layer1_size=64,
        layer2_size=64,
        batch_size=32,
        alpha=0.5,
    ):
        self.name = "SAC"

        assert isinstance(action_space, gym.spaces.Box)
        self.state_space = state_space
        self.action_space = action_space

        self.action_low = torch.Tensor(action_space.low)
        self.action_high = torch.Tensor(action_space.high)

        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        state_size = state_space.shape[0]
        nb_actions = self.action_space.shape[0]

        self.actor = (
            torch.nn.Sequential(
                torch.nn.Linear(state_size, layer1_size),
                torch.nn.ReLU(),
                torch.nn.Linear(layer1_size, layer2_size),
                torch.nn.ReLU(),
                torch.nn.Linear(layer2_size, 2 * nb_actions),
            )
            .to(device)
            .float()
        )
        self.critic = (
            torch.nn.Sequential(
                torch.nn.Linear(state_size + nb_actions, layer1_size),
                torch.nn.ReLU(),
                torch.nn.Linear(layer1_size, layer2_size),
                torch.nn.ReLU(),
                torch.nn.Linear(layer2_size, 1),
            )
            .to(device)
            .float()
        )

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.target_critic = deepcopy(self.critic)

        self.alpha = alpha

    def update_towards(self, model, other_model, tau=None):
        """Polyak update towards `other_model`."""
        if tau is None:
            tau = self.tau
        for self_param, other_param in zip(model.parameters(), other_model.parameters()):
            self_param.data.copy_(self_param.data * (1.0 - tau) + other_param.data * tau)

    def sample_action(self, state, reparameterize=False):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(device)

        if len(state.shape) == 1:  # Add batch dimension
            state = state.unsqueeze(0)

        # Get actor outputs
        actor_output = self.actor(state).view(state.shape[0], -1, 2)

        # Compute action mean and std
        actions_means = actor_output[:, 0]
        actions_log_stds = torch.clamp(actor_output[:, 1], -20, 2)  # Clamp for stability
        actions_stds = torch.exp(actions_log_stds)

        # Get action distribution (cf. torch.distributions)
        actions_distribution = torch.distributions.normal.Normal(
            actions_means,
            actions_stds,
        )

        # Sample from the action distribution using the reparametrization trick if necessary
        if reparameterize:
            actions = actions_distribution.rsample()
        else:
            actions = actions_distribution.sample()

        # Bound actions to [-1, 1]
        bounded_actions = torch.tanh(actions)

        # Compute log probabilities (/!\ Don't forget the transformation term!)
        log_probs = actions_distribution.log_prob(actions)
        log_probs -= torch.log(1 - bounded_actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1)

        # Scale actions to the environment bounds
        scaled_actions = (bounded_actions + 1) / 2  # In [0, 1]
        scaled_actions = (self.action_high - self.action_low) * scaled_actions + self.action_low  # In [low, high]

        return scaled_actions, log_probs

    def get_action(self, state):
        actions, _ = self.sample_action(state, reparameterize=False)
        return actions.squeeze().cpu().detach().numpy()

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, done = self.replay_buffer.sample(
            self.batch_size
        )

        with torch.no_grad():
            # Compute a' = \pi(s') and log \pi(s')
            next_actions, next_log_probs = self.sample_action(next_states)

            # Compute Q(s', \pi(s'))
            next_q_values = self.target_critic(
                torch.cat((next_states, next_actions), -1)
            ).view(-1)

        # Compute target = r + \gamma * (1 - d) * [ Q(s', \pi(s')) - \alpha log \pi(s') ]
        td_error = rewards + self.gamma * (1 - done) * (
            next_q_values - self.alpha * next_log_probs
        )

        # Compute Q(s, a)
        q_values = self.critic(torch.cat((states, actions), 1)).view(-1)

        # Compute critic loss (TD error)
        self.critic_optimizer.zero_grad()
        critic_loss = torch.nn.functional.mse_loss(q_values, td_error)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.update_towards(self.target_critic, self.critic)

        # Train actor: compute Q(s, \pi(s))
        actions, log_probs = self.sample_action(states, reparameterize=True)
        critic_values = self.critic(torch.cat((states, actions), -1)).view(-1)

        # Compute soft actor loss
        self.actor_optimizer.zero_grad()
        actor_loss = (self.alpha * log_probs.view(-1) - critic_values).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
