import gym
import torch


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    """
    An global agent class that describe the interactions between our agent and it's environment
    """

    def __init__(self, state_space, action_space, device=DEFAULT_DEVICE, name="base_agent"):
        self.name = name  # The name is used inside plot legend, outputs directory path, and outputs file names

        self.state_space = state_space
        self.state_shape = state_space.shape
        self.state_size = state_space.shape[0]  # Assume state space is continuous

        self.continuous = isinstance(action_space, gym.spaces.Box)
        self.action_space = action_space
        self.nb_actions = self.action_space.shape[0] if self.continuous else self.action_space.n
        self.last_state = None  # Useful to store interaction when we receive (new_stare, reward, done) tuple
        self.device = device
        self.episode_id = 0
        self.episode_time_step_id = 0
        self.time_step_id = 0

    def on_simulation_start(self):
        """
        Called when an episode is started. will be used by child class.
        """
        pass

    def on_episode_start(self, state):
        self.last_state = state
        self.episode_time_step_id = 0
        self.episode_id = 0

    def action(self, state):
        res = self.action_space.sample()
        return res

    def on_action_stop(self, action, new_state, reward, done):
        self.episode_time_step_id += 1
        self.time_step_id += 1
        self.last_state = new_state

    def on_episode_stop(self):
        self.episode_id += 1

    def on_simulation_stop(self):
        pass
