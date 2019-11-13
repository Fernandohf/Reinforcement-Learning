import torch
from unityagents import UnityEnvironment
from utilities import train_MADDPG
from maddpg import MADDPG


env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# Hyperparameters
PARAMETERS = {
    'BUFFER_SIZE': int(1e4),           # Replay buffer size
    'BATCH_SIZE': 256,                 # Minibatch size
    'GAMMA': 1.,                      # Discount factor
    'TAU': 2e-3,                       # Soft update of target parameters
    'UPDATE_EVERY': 1,                 # Wait for more experiences before update

    'N_AGENTS': 2,                     # Total number of agents
    'STATE_SIZE': 24,                  # Size of the state for each agent
    'ACTION_SIZE': 2,                  # Size of actions for each agent

    'ACTOR_LR': 1e-2,                  # Learning rate of the actor
    'ACTOR_WEIGHT_DECAY': 0.000,       # Actor L2 weight decay
    'ACTOR_GRADIENT_CLIP_VALUE': 2,    # Max gradient modulus for clipping

    'CRITIC_LR': 1e-2,                 # Learning rate of the critic
    'CRITIC_WEIGHT_DECAY': 0.000,      # Critic L2 weight decay
    'CRITIC_GRADIENT_CLIP_VALUE': 1,   # Max gradient modulus for clipping

    'NOISE_TYPE': 'normal',            # Type of noise used: 'normal' or 'ou'
    'N_SIGMA': .3,                     # Normal noise sigma parameters
    'N_MEAN': 0.,
    'N_EPS_BETA': .01,
    'N_EPS_INIT': 1,
    'N_EPS_MIN': .001,

    'OU_THETA': 1e-3,                  # OU noise theta parameter
    'OU_SIGMA': 1e-3,                  # OU noise sigma parameters

    'SEED': 42,                       # Random seed
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

agent = MADDPG(PARAMETERS)

# Hyperparameters
N_EPISODES = 1000
MAX_T = 1000
SUCCESS_SCORE = .5


scores = train_MADDPG(env, agent, n_episodes=N_EPISODES, max_t=MAX_T, success_score=SUCCESS_SCORE,
                      print_every=100)
