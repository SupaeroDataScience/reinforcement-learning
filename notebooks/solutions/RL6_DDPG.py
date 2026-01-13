# To get an example of implementation, uncomment the line above

from copy import deepcopy

import gymnasium as gym
import torch


class DDPGAgent:
    def __init__(
        self,
        state_space,
        action_space,
        actor_lr=0.000025,
        critic_lr=0.00025,
        tau=0.001,
        gamma=0.99,
        buffer_size=1000000,
        layer1_size=200,
        layer2_size=150,
        batch_size=64,
        noise_std=0.1,
    ):
        self.name = "DDPG"

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

        # Noise distribution to add to the action for exploration
        self.noise_distribution = torch.distributions.normal.Normal(
            torch.zeros(nb_actions),
            noise_std * torch.ones(nb_actions),
        )

        # Define the actor and critic networks
        self.actor = DefaultNN(
            state_size,
            layer1_size,
            layer2_size,
            nb_actions,
            actor_lr,
            device,
            last_activation=torch.tanh,
        )
        self.critic = DefaultNN(
            state_size + nb_actions,
            layer1_size,
            layer2_size,
            1,
            critic_lr,
            device,
        )

        # Define the corresponding target networsk
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(device)

        # Get action from the actor
        with torch.no_grad():
            action = self.actor(state).to(device)

        # Add noise to action while making sure it stays valid
        noise = self.noise_distribution.sample()
        action = torch.clamp(action + noise, self.action_low, self.action_high)

        return action.cpu().detach().numpy()

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Compute Q(s, a)
        critic_value = self.critic(torch.concat((states, actions), dim=-1))

        with torch.no_grad():
            # Compute \mu(s')
            target_actions = self.target_actor(next_states)

            # Compute Q(s', \mu(s')), which should approximate max_a' Q(s', a')
            next_critic_values = self.target_critic(
                torch.concat((next_states, target_actions), dim=-1)
            ).squeeze()

        # Target r + \gamma * (1 - done) * Q(s', \mu(s'))
        target = torch.addcmul(
            rewards, 1 - dones, next_critic_values, value=self.gamma
        ).view(self.batch_size, 1)

        # Compute critic loss as in DQN (=TD error)
        critic_loss = torch.nn.functional.mse_loss(target, critic_value)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Update actor in order to maximize Q(s, \mu(s))
        actor_actions = self.actor(states)
        actor_loss = -self.critic(torch.concat((states, actor_actions), dim=-1))
        actor_loss = torch.mean(actor_loss)

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.target_critic.update_towards(self.critic, tau=self.tau)
        self.target_actor.update_towards(self.actor, tau=self.tau)
