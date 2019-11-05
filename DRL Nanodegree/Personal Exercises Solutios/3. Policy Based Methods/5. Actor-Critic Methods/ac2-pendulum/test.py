from subprocess_env import SubprocVecEnv
from a2c_agent import Agent, train_a2c
from utils import make_env, n_step_boostrap


# Hyperparameters
ENV_NAME = 'Pendulum-v0'
ENV_SEED = 42
N_ENVS = 6
N_STEP_BOOSTRAP = 4


def main():
    # Create envs
    env_fns = [make_env(ENV_NAME, ENV_SEED + i) for i in range(N_ENVS)]

    # Multiprocessing Environments
    mp_envs = SubprocVecEnv(env_fns)

    agent = Agent(state_size=3, action_size=1, actor_hidden_size=(8,),
                  critic_hidden_size=(8,), random_seed=2)
    # initial_states = mp_envs.reset()
    # S, A, R, Sp, dones = n_step_boostrap(mp_envs, agent, initial_states, 4)
    # agent.learn(S, A, R, Sp, dones)
    train_a2c(mp_envs, agent, episodes=100)


if __name__ == "__main__":
    main()
