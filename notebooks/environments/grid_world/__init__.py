from environments.grid_world.discrete_grid_world import DiscreteGridWorld
from environments.grid_world.goal_conditioned_discrete_grid_world import GoalConditionedDiscreteGridWorld
import gym
import warnings
from gym.envs.registration import register


def register_environment(env_id, env_entry_point):
    """
    Register an environment with the given id.
    Make sure the environment is registered only once to prevent 'cannot re-register id' errors
    """
    env_dict = gym.envs.registration.registry.env_specs.copy()
    for env in env_dict:
        if env_id == env:
            warnings.warn("Updating environment {} with a new registration.".format(env))
            del gym.envs.registration.registry.env_specs[env]

    register(
        id=env_id,
        entry_point=env_entry_point,
    )
    

register_environment(
    env_id='discrete_grid_world-v0',
    env_entry_point='environments.grid_world.discrete_grid_world:DiscreteGridWorld'
)

register_environment(
    env_id='goal_conditioned_discrete_grid_world-v0',
    env_entry_point='environments.grid_world.goal_conditioned_discrete_grid_world:GoalConditionedDiscreteGridWorld'
)
