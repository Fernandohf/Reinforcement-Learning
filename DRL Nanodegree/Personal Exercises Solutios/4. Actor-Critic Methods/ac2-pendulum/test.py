import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from utils import n_step_boostrap
from subprocess_env import SubprocVecEnv
from a2c_agent import Agent
from utils import make_env


# Hyperparameters
ENV_NAME = 'Pendulum-v0'
ENV_SEED = 42
N_ENVS = 2


def main():
    # Create envs
    env_fns = [make_env(ENV_NAME, ENV_SEED) for _ in range(N_ENVS)]

    # Multiprocessing Environments
    mp_envs = SubprocVecEnv(env_fns)

    agent = Agent(state_size=3, action_size=1, actor_hidden_size=(8,),
                  critic_hidden_size=(8,), random_seed=2)

    initial_states = mp_envs.reset()
    n_step_boostrap(mp_envs, agent, initial_states, n=5)


if __name__ == "__main__":
    main()
