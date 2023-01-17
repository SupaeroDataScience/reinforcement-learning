import gym


default_device = torch.device("cuda" if torch.cuda.is_availablle() else "cpu")


class GoalConditionedAgent:
    """
    An global agent class for goal conditioned agents. The # NEW tag indicate differences between Agent class and this
    one.
    """

    def __init__(self, state_space, action_space, device=default_device, name="Random agent"):
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

        # New
        # The goal we are trying to reach will be set when a new episode is started.
        # We need to keep it in memory during the entire episode
        self.current_goal = None

    def on_simulation_start(self):
        """
        Called when an episode is started. will be used by child class.
        """
        pass

    def on_episode_start(self, state, goal):
        self.last_state = state
        self.episode_time_step_id = 0
        self.episode_id = 0
        self.current_goal = goal  # NEW

    def action(self, state):  # NEW, this function should now take the goal in consideration
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
