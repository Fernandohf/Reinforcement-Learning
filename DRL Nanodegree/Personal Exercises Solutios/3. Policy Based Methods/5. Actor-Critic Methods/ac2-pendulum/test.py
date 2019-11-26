from a2c_agent import A2CAgent, train_a2c
from utils import n_step_boostrap, make_multi_envs


# Hyperparameters
ENV_NAME = 'Pendulum-v0'
ENV_SEED = 42
N_ENVS = 12
N_STEP_BOOSTRAP = 5


def main():
    # Multiprocessing Environments
    mp_envs = make_multi_envs(N_ENVS, ENV_NAME, ENV_SEED)

    agent = A2CAgent(state_size=3, action_size=1, actor_hidden_size=128,
                     critic_hidden_size=(128, 32), random_seed=2)
    # initial_states = mp_envs.reset()
    # S, A, R, Sp, dones = n_step_boostrap(mp_envs, agent, initial_states, N_STEP_BOOSTRAP)
    # pass
    # agent.learn(S, A, R, Sp, dones)
    train_a2c(mp_envs, agent, episodes=100, n_step=N_STEP_BOOSTRAP)


if __name__ == "__main__":
    main()
