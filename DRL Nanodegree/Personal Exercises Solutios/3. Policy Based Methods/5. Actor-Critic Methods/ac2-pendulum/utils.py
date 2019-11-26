"""
Utilities functions
"""
import numpy as np
import gym
from subprocess_env import SubprocVecEnv


def make_env(env_name, seed):
    """
    Auxiliary function used to create the GYM environment with the given seed.

    Parameters
    ----------
    env_name: string
        Name of the GYM environment
    seed: int
        Seed for the environment
    Return
    ------
    _init: func
        function that creates env
    """
    def _init():
        env = gym.make(env_name)
        env.seed(seed)
        return env
    return _init


def make_multi_envs(n_envs, env_name, seed):
    """
    Create multiple parallel GYM environments.

    Parameters
    ----------
    n_env: int
        Number of parallel environments
    env_name: string
        Name of the GYM environment
    seed: int
        Seed for the environment

    Return
    ------
    mp_envs: SubprocVecEnv
        Parallel environments
    """
    # Create envs
    env_fns = [make_env(env_name, seed + i) for i in range(n_envs)]

    # Multiprocessing Environments
    mp_envs = SubprocVecEnv(env_fns)

    return mp_envs


# Collect n_step bootstrap
def n_step_boostrap(envs, agent, previous_states, n_bootstrap=5):
    """
    Perform n_step bootstrap on the list of parallel environments.

    Parameters:
    -----------
    envs: parallelEnv
        List of environments running on parallel
    agent: Agent with act function
        Policy being used to get the S.A.R.S'. sequences
    previous_states: tuple
        Previous states of the environments
    n_bootstrap: int
        Number of steps used

    Returns:
    -------
    experiences: tuple
        Experience for each parallel environment
        It has the format:
         n_envs, n, (states, actions, rewards, next_states)
    """

    # Initialize returning lists and start the game!
    state_list = []
    reward_list = []
    action_list = []
    states_next_list = []
    dones_list = []

    for t in range(n_bootstrap):

        # Actions from states for each env
        actions_env = agent.act(previous_states, add_noise=True)

        # Advance the environment
        states_next_env, rewards_env, done_envs, _ = envs.step(actions_env)

        # Store the result
        state_list.append(previous_states)
        reward_list.append(rewards_env.reshape(-1, 1))
        action_list.append(actions_env)
        states_next_list.append(states_next_env)
        dones_list.append(done_envs.reshape(-1, 1))

        previous_states = states_next_env

        # Stop if any of the trajectories is done
        # we want all the lists to be retangular
        if done_envs.any():
            break

    # Return states, actions, rewards, states_next
    experiences = (np.stack(state_list, axis=1), np.stack(action_list, axis=1),
                   np.stack(reward_list, axis=1), np.stack(states_next_list, axis=1),
                   np.stack(dones_list, axis=1))
    return experiences
