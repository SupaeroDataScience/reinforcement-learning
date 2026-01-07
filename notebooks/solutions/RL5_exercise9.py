### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

import numpy as np
import torch
from copy import deepcopy

class DQNAgent:
    def __init__(self, config, model):
        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]
        self.nb_actions = config["nb_actions"]
        self.memory = ReplayBuffer(config["buffer_size"])
        self.epsilon_max = config["epsilon_max"]
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_stop = config["epsilon_decay_period"]
        self.epsilon_delay = config["epsilon_delay_decay"]
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop
        self.nb_gradient_steps = config["gradient_steps"]
        self.nb_trials = config["nb_trials"]  # NEW NEW NEW
        self.total_steps = 0
        self.model = model
        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config["learning_rate"]
        )
        self.target_model = deepcopy(self.model).to(device)
        self.update_target_freq = config["update_target_freq"]

    def MC_eval(self, env, nb_trials):  # NEW NEW NEW
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            s, _ = env.reset()
            done = False
            total_reward = 0
            discounted_reward = 0
            discount_factor = 1
            while not done:
                a = greedy_action(self.model, s)
                s, r, done, _, _ = env.step(a)
                total_reward += r
                discounted_reward += discount_factor * r
                discount_factor *= self.gamma
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)

    def eval_init_state(self, env, nb_trials):  # NEW NEW NEW
        with torch.no_grad():
            val = []
            for _ in range(nb_trials):
                s, _ = env.reset()
                val.append(
                    self.model(torch.Tensor(s).unsqueeze(0).to(device)).max().item()
                )
        return np.mean(val)

    def gradient_step(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample from memory
        X, A, R, Y, D = self.memory.sample(self.batch_size)

        # Compute y_t = r(s, a) + \gamma * max_a' Q(s', a')
        QYmax = self.target_model(Y).max(1)[0].detach()
        update = torch.addcmul(R, 1 - D, QYmax, value=self.gamma)

        # Compute loss
        QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
        loss = self.criterion(QXA, update.unsqueeze(1))

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env, max_episode):
        # Training monitoring
        episode_return = []
        MC_avg_total_reward = []  # NEW NEW NEW
        MC_avg_discounted_reward = []  # NEW NEW NEW
        avg_Q_init_state = []  # NEW NE W NEW

        # Initialization
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0

        while episode < max_episode:
            # Update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)

            # Select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = np.random.randint(self.nb_actions)
            else:
                action = greedy_action(self.model, state)

            # Step and store in memory
            next_state, reward, done, _, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # Train
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()

            # Update target network if needed
            if step % self.update_target_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            # Next transition
            step += 1
            if done:
                episode += 1

                # Monitoring
                if self.nb_trials > 0:  # NEW NEW NEW
                    MC_dr, MC_tr = self.MC_eval(env, self.nb_trials)
                    Q0 = self.eval_init_state(env, self.nb_trials)
                    MC_avg_total_reward.append(MC_tr)
                    MC_avg_discounted_reward.append(MC_dr)
                    avg_Q_init_state.append(Q0)
                    episode_return.append(episode_cum_reward)
                    print(
                        f"Episode {episode:2d}, epsilon {epsilon:6.2f},",
                        f"batch size {len(self.memory):4d}, ep return {episode_cum_reward:4.1f},",
                        f"MC tot {MC_tr:6.2f}, MC disc {MC_dr:6.2f}, Q0 {Q0:6.2f}",
                    )
                else:
                    episode_return.append(episode_cum_reward)
                    print(
                        f"Episode {episode:3d}, epsilon {epsilon:6.2f},",
                        f"batch size {len(self.memory):5d},",
                        f"episode return {episode_cum_reward:4.1f}",
                    )

                # Start new episode
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state

        return (
            episode_return,
            MC_avg_discounted_reward,
            MC_avg_total_reward,
            avg_Q_init_state,
        )
