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

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]

print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# Hyperparameters
PARAMETERS = {
    'BUFFER_SIZE': int(1e6),           # Replay buffer size
    'BATCH_SIZE': 256,                 # Minibatch size
    'GAMMA': 1.0,                      # Discount factor
    'TAU': 1e-2,                       # Soft update of target parameters
    'UPDATE_EVERY': 1,                 # Wait for more experiences before update
    'REPLAY_ALPHA': .1,                # 1 = full prioritization, 0 = no prioritization

    'N_AGENTS': 2,                     # Total number of agents
    'STATE_SIZE': 24,                  # Size of the state for each agent
    'ACTION_SIZE': 2,                  # Size of actions for each agent

    'ACTOR_LR': 5e-4,                  # Learning rate of the actor
    'ACTOR_WEIGHT_DECAY': 0.0000,      # Actor L2 weight decay
    'ACTOR_GRADIENT_CLIP_VALUE': 1,    # Max gradient modulus for clipping

    'CRITIC_LR': 1e-2,                 # Learning rate of the critic
    'CRITIC_WEIGHT_DECAY': 0.001,      # Critic L2 weight decay
    'CRITIC_GRADIENT_CLIP_VALUE': 1,   # Max gradient modulus for clipping

    'NOISE_TYPE': 'ou',                # Type of noise used: 'normal' or 'ou'
    'N_SIGMA': .3,                     # Normal noise sigma parameters

    'OU_THETA': .001,                   # OU noise theta parameter
    'OU_SIGMA': .001,                   # OU noise sigma parameters

    'SEED': 42,                       # Random seed
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

agent = MADDPG(PARAMETERS)

# Hyperparameters
N_EPISODES = 2000
MAX_T = 1000
SUCCESS_SCORE = .5


scores = train_MADDPG(env, agent, n_episodes=N_EPISODES, max_t=MAX_T, success_score=SUCCESS_SCORE,
                      print_every=100)
