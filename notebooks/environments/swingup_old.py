import gym
from gym import logger
import numpy as np
from gym.spaces import Box


class CartPoleSwingUp(gym.Wrapper):
    def __init__(self, env=gym.make('CartPole-v1'), **kwargs):
        super(CartPoleSwingUp, self).__init__(env, **kwargs)
        self.theta_dot_threshold = 4*np.pi

    def reset(self):
        self.env.env.state = [0, 0, np.pi, 0] + super().reset()
        return np.array(self.env.env.state)

    def step(self, action):
        state, reward, done, _ = super().step(action)
        self.env.env.steps_beyond_done = None
        x, x_dot, theta, theta_dot = state
        theta = (theta+np.pi)%(2*np.pi)-np.pi
        self.env.env.state = [x, x_dot, theta, theta_dot]
        
        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta_dot < -self.theta_dot_threshold \
               or theta_dot > self.theta_dot_threshold
        
        if done:
            # game over
            reward = -10.
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
            elif self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
                self.steps_beyond_done += 1
        else:
            if -self.theta_threshold_radians < theta and theta < self.theta_threshold_radians:
                # pole upright
                reward = 1.
            else:
                # pole swinging
                reward = 0.

        return np.array([x, x_dot, theta, theta_dot]), reward, done, {}



class GoalCartPoleSwingUp(CartPoleSwingUp):
    """
    A goal conditioned version of CartPoleSwingUp.
    At each episode, the environment will give a random goal to the agent.
    """
    def __init__(self, env=gym.make('CartPole-v1'), **kwargs):
        super().__init__(env=env, **kwargs)
        
        # Initialise goal generation
        low = np.array([-np.pi, -1])
        high = np.array([np.pi, 1])
        self.goal_generator = Box(low, high)
        
        self.goal = None
        self.reset_goal()
        self.max_iterations = 200
        self.interaction_id = 0
       
    def filter_state(self, state: np.ndarray):
        return state[-2:]

    def reset_goal(self, test=False) -> np.ndarray:
        """
        Choose a goal for the agent.
        test = True mean that we want to test our agent's abilities to bring the pole straight up, so we should choose this
        specifically as a goal.
        :return: the goal
        """
        self.goal = self.goal_generator.sample()
        if test:
            self.goal[0] = 0.  # Angle of the poll
            self.goal[1] = 0.  # Angle dot of the poll
        return self.goal
        
    def reset(self, test=False):
        self.env.env.state = [0, 0, np.pi, 0] + super().reset()
        self.interaction_id = 0
        return np.array(self.env.env.state), self.reset_goal(test=test)
    
    def reached(self, state, goal):
        theta = state[2] - goal[0]
        theta_dot = state[3] - goal[1]
        
        theta_reached = - self.theta_threshold_radians < theta \
            and theta < self.theta_threshold_radians
        
        dot_reached = - self.theta_dot_threshold < theta_dot \
            and theta_dot < self.theta_dot_threshold
        
        return theta_reached and dot_reached

    def step(self, action):
        state, reward, done, _ = super().step(action)
        self.env.env.steps_beyond_done = None
        x, x_dot, theta, theta_dot = state
        theta = (theta+np.pi)%(2*np.pi)-np.pi
        self.env.env.state = [x, x_dot, theta, theta_dot]
        
        reached = self.reached(state, self.goal)
        self.interaction_id += 1
        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta_dot < -self.theta_dot_threshold \
               or theta_dot > self.theta_dot_threshold \
               or self.interaction_id >= self.max_iterations
        
        if done:
            # game over
            reward = -10.
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
            elif self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
                self.steps_beyond_done += 1
        elif reached:
            reward = 1.
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
            elif self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
                self.steps_beyond_done += 1
                done = True
        else:
            """ 
            This part of the code should be removed because we want to reach goals. If we can have reward by ignoring the 
            goal, the agent will not reach it.
            
            if -self.theta_threshold_radians < theta and theta < self.theta_threshold_radians:
                # pole upright
                reward = 1.
            else:
                # pole swinging
                reward = 0.
            """
            reward = 0.

        return np.array([x, x_dot, theta, theta_dot]), reward, done, {}
