"""
Utilities functions
"""
import numpy as np
import gym


def make_env(env_name, seed):
    def _init():
        env = gym.make(env_name)
        env.seed(seed)
        return env
    return _init


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
        reward_list.append(rewards_env)
        action_list.append(actions_env)
        states_next_list.append(states_next_env)
        dones_list.append(done_envs)

        previous_states = states_next_env

        # Stop if any of the trajectories is done
        # we want all the lists to be retangular
        if done_envs.any():
            break

    # Return states, actions, rewards, states_next
    experiences = (np.dstack(state_list), np.dstack(action_list),
                   np.vstack(reward_list).T, np.dstack(states_next_list),
                   np.vstack(dones_list).T)
    return experiences
