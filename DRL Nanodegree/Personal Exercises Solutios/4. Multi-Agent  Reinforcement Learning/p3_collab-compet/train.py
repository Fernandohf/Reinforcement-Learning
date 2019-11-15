import torch
from unityagents import UnityEnvironment
from utilities import train_MADDPG
from maddpg import MADDPG
import wandb

# Load environment
env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")

# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# Default Hyperparameters
PARAMETERS = {
    'BUFFER_SIZE': int(1e6),           # Replay buffer size
    'BATCH_SIZE': 256,                 # Minibatch size
    'GAMMA': .99,                      # Discount factor
    'TAU': 5e-3,                       # Soft update of target parameters
    'UPDATE_EVERY': 1,                 # Wait for more experiences before update

    'N_AGENTS': 2,                     # Total number of agents
    'STATE_SIZE': 24,                  # Size of the state for each agent
    'ACTION_SIZE': 2,                  # Size of actions for each agent

    'ACTOR_LR': 1e-3,                  # Learning rate of the actor
    'ACTOR_WEIGHT_DECAY': 0.000,       # Actor L2 weight decay
    'ACTOR_GRADIENT_CLIP_VALUE': 5,    # Max gradient modulus for clipping

    'CRITIC_LR': 1e-3,                 # Learning rate of the critic
    'CRITIC_WEIGHT_DECAY': 0.00001,    # Critic L2 weight decay
    'CRITIC_GRADIENT_CLIP_VALUE': 1,   # Max gradient modulus for clipping

    'NOISE_TYPE': 'normal',            # Type of noise used: 'normal' or 'ou'
    'N_SIGMA': .4,                     # Normal noise sigma parameters
    'N_MEAN': 0.,                      # Normal noise mu parameters
    'N_EPS_BETA': .1,                  # Normal noise decay rate
    'N_EPS_INIT': 1,                   # Normal noise initial decay value
    'N_EPS_MIN': .01,                  # Normal noise min decay value

    'OU_THETA': 1e-2,                  # OU noise theta parameter
    'OU_SIGMA': 1e-2,                  # OU noise sigma parameters

    'SEED': 42,                       # Random seed
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    # Training Hyperparameters
    'N_EPISODES': 2000,
    'MAX_T': 2000,
    'SUCCESS_SCORE': .5,
    'PRINT_EVERY': 100,

    # Save on W&B
    'WANDB': True,
}

if PARAMETERS['WANDB']:
    # Save on wandb
    wandb.init(project="maddpg", config=PARAMETERS)

# Agent
agent = MADDPG(PARAMETERS)

# Training job
scores = train_MADDPG(env, agent, n_episodes=PARAMETERS['N_EPISODES'],
                      max_t=PARAMETERS['MAX_T'], success_score=PARAMETERS['SUCCESS_SCORE'],
                      print_every=PARAMETERS['PRINT_EVERY'], brain_name=brain_name,
                      use_wandb=PARAMETERS['WANDB'])
