# Multi Agent Reinforcement Learning

This project implements the MADDPG algorithm to solve the [Unity Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

## The Tennis Environment

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

- **Observation Space**: The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket for each agent.
- **Action Space**: Two continuous actions per agent are available, corresponding to movement toward (or away from) the net, and jumping.
- **Solution**: In order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

## The Algorithm

The algorithm used to solve the environment is called [Multi Agent Deep Deterministic Policy Gradients](https://arxiv.org/pdf/1706.02275.pdf). It is a multi agent version of the [DDPG](https://arxiv.org/pdf/1509.02971.pdf), which is suitable for continuous control scenarios (such as this Tennis task). Similarly to its single agent counterpart, **MADDPG** is an off-policy, actor-critic algorithm that uses the concept of local and target networks. Therefore, it uses a **Replay Buffer** to reduce consecutive correlations biases, as well as **Soft Updates** to stabilize training.

The main difference is that it uses the framework of **Centralized Training with Decentralized Execution**. Briefly, it consists of using the maximum amount of information during training but restricting it during testing time. In the **MADDPG**, this implies that the critic networks for each agents has access to other agents actions and states during training. While in execution time, only actors are present, therefore, they can only see their agents states and actions.

## Methodology and Hyperparameters

The algorithm was executed for **2000** episodes. As soon as the conditions for solving the environment were met, the networks weights were saved, but the algorithm continued to see if further improvements could be achieved. The algorithm results were recorded using the [Weight and Biases](https://app.wandb.ai/fernandohf/maddpg) tool. It records, hyperparameters, losses, network weights, system information and  more(check the link for more information). All the hyperparameters with their description are available on the file `config.yml` and shown below.

```yaml
wandb_version: 1
ACTION_SIZE:
  desc: Size of actions for each agent
  value: 2
ACTOR_GRADIENT_CLIP_VALUE:
  desc: Max gradient modulus for clipping
  value: 5
ACTOR_LR:
  desc: Learning rate of the actor
  value: 0.001
ACTOR_WEIGHT_DECAY:
  desc: Actor L2 weight decay
  value: 0.0
BATCH_SIZE:
  desc: Mini-batch size
  value: 256
BUFFER_SIZE:
  desc: Replay buffer size
  value: 1000000
CRITIC_GRADIENT_CLIP_VALUE:
  desc: Max gradient modulus for clipping
  value: 1
CRITIC_LR:
  desc: Learning rate of the critic
  value: 0.001
CRITIC_WEIGHT_DECAY:
  desc: Critic L2 weight decay
  value: 1.0e-05
DEVICE:
  desc: Device being trained (cuda/cpu)
  value: cuda
GAMMA:
  desc: Discount factor
  value: 0.99
MAX_T:
  desc: Maximum number of time steps
  value: 2000
NOISE_TYPE:
  desc: Type of noise used (normal/ou)
  value: normal
N_AGENTS:
  desc: Total number of agents
  value: 2
N_EPISODES:
  desc: Number of episodes to run the algorithm
  value: 2000
N_EPS_BETA:
  desc: Normal noise decay rate
  value: 0.1
N_EPS_INIT:
  desc: Normal noise initial decay value
  value: 1
N_EPS_MIN:
  desc: Normal noise min decay value
  value: 0.01
N_MEAN:
  desc: Normal noise mu parameters
  value: 0.0
N_SIGMA:
  desc: Normal noise sigma parameters
  value: 0.4
OU_SIGMA:
  desc: OU noise sigma parameter
  value: 0.01
OU_THETA:
  desc: OU noise theta parameter
  value: 0.01
PRINT_EVERY:
  desc: Reporting interval
  value: 100
SEED:
  desc: Random seed
  value: 42
STATE_SIZE:
  desc: Size of the state for each agent
  value: 24
SUCCESS_SCORE:
  desc: Score to save the model and solve the environment
  value: 0.5
TAU:
  desc: Soft update of target parameters
  value: 0.005
UPDATE_EVERY:
  desc: Wait for more experiences before update
  value: 1
WANDB:
  desc: Wether or not save results on W&B
  value: true
_wandb:
  desc: null
  value:
    cli_version: 0.8.15
    framework: torch
    is_jupyter_run: false
    python_version: 3.6.9
```

## Results

The model achieved the solved condition (average rewards above **0.5** on last 100 episodes) at episode **929** and continued improving until episode **1000** with average score around **0.9**. From there, the performance dropped a little and then started increasing to achieve the highest during this training session, **1.5** at episode **1169**. From that point forwards, the performance only dropped.

The main training graphs are plotted below and are also available on the [W&B project site](https://app.wandb.ai/fernandohf/maddpg).

### Plots

#### Average Rewards last 100 Episodes

![Average Rewards plot]( "Average Rewards Plot")

#### Smoothed Scores

![Average Rewards plot]( "Average Rewards Plot")

### Trained Agents

The gif below is a recording of the trained agent.

```python
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
```

### 2. Examine the State and Action Spaces

In this environment, a double-jointed arm can move to target locations. A reward of `+0.1` is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of `33` variables corresponding to position, rotation, velocity, and angular velocities of the arm.  Each action is a vector with four numbers, corresponding to torque applicable to two joints.  Every entry in the action vector must be a number between `-1` and `1`.

Run the code cell below to print some information about the environment.


```python
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
```

    Number of agents: 2
    Size of each action: 2
    There are 2 agents. Each observes a state with length: 24
    The state for the first agent looks like: [ 0.          0.          0.          0.          0.          0.
      1.          0.          0.          0.          0.          0.
      2.          0.          0.          0.         -6.65278625 -1.5
     -0.          0.          6.83172083  6.         -0.          0.        ]


### 4. Reinforcement Learning Algorithm

The algorityhm used is the DDPG Agent as described by the [DDPG Paper](https://arxiv.org/pdf/1509.02971) and the code provided on [Udacity's DRLND Github Repository](https://github.com/udacity/deep-reinforcement-learning#dependencies). The Agent combined a policy based and values based approach, known as Actor-Critic method.

This algorithm uses an Actor-Critic approach, which the Critic is used to approximate the maximizer over the "Q-values" of the next state and not a learned baseliner (such as A2C). The Actor learns a deterministic policy to select the best action given the states. the algorithm has two interesting aspects:
- **Soft Updates**: Instead of updating the target/regular networks weights every `C` timesteps, weights are gradually mixed (generally 0.01-0.001%) at every timestep.
- **Replay  Buffer**: Samples thes sequences of S.A.R.S' randomly, therefore reducing the high correlations between these sequences.


All these logic are implemented on the files `model.py` and `ddpg_agent.py`.
Some hyperparameters are also listed on the file `ddpg_agent.py`

### 3. MADDPG Agent

The next cell implements the DDPG Agent on the Unity Environment.


```python
# Hyperparameters
PARAMETERS = {
    'BUFFER_SIZE': int(1e7),          # Replay buffer size
    'BATCH_SIZE': 256,                # Minibatch size
    'GAMMA': 1.,                      # Discount factor
    'TAU': 1e-2,                      # Soft update of target parameters
    'UPDATE_EVERY': 1,                # Wait for more experiences before update

    'N_AGENTS': 2,                    # Total number of agents
    'STATE_SIZE': 24,                 # Size of the state for each agent
    'ACTION_SIZE': 2,                 # Size of actions for each agent

    'ACTOR_LR': 1e-3,                 # Learning rate of the actor
    'ACTOR_WEIGHT_DECAY': 0.0000,     # Actor L2 weight decay
    'ACTOR_GRADIENT_CLIP_VALUE': 2,   # Max gradient modulus for clipping

    'CRITIC_LR': 1e-3,                # Learning rate of the critic
    'CRITIC_WEIGHT_DECAY': 0.0001,    # Critic L2 weight decay
    'CRITIC_GRADIENT_CLIP_VALUE': 5,  # Max gradient modulus for clipping

    'NOISE_TYPE': 'ou',               # Type of noise used: 'normal' or 'ou'
    'N_SIGMA': .4,                    # Normal noise sigma parameters

    'OU_THETA': .002,                   # OU noise theta parameter
    'OU_SIGMA': .002,                  # OU noise sigma parameters

    'SEED': 42,                       # Random seed
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
```


```python
# Load and define agent
from maddpg import MADDPG

agent = MADDPG(PARAMETERS)
```


```python
# Hyperparameters
N_EPISODES = 1000
MAX_T = 1000
SUCCESS_SCORE = .5
```


```python
from utilities import train_MADDPG

scores = train_MADDPG(env, agent, n_episodes=N_EPISODES, max_t=MAX_T,
                      success_score=SUCCESS_SCORE, print_every=100)
```

    Episode 100/1000	Average Score (last 100): 0.0029	 Last score: 0.0000
    Episode 200/1000	Average Score (last 100): 0.0038	 Last score: 0.0000
    Episode 300/1000	Average Score (last 100): 0.0068	 Last score: 0.0000
    Episode 400/1000	Average Score (last 100): 0.0020	 Last score: 0.0000
    Episode 455/1000	Average Score (last 100): 0.0030	 Last score: 0.0000


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-12-6316e8a3e877> in <module>
          2
          3 scores = train_MADDPG(env, agent, n_episodes=N_EPISODES, max_t=MAX_T,
    ----> 4                       success_score=SUCCESS_SCORE, print_every=100)


    D:\ARQUIVOS PESSOAIS\GitHub\Reinforcement-Learning\DRL Nanodegree\Personal Exercises Solutios\4. Multi-Agent  Reinforcement Learning\p3_collab-compet\utilities.py in train_MADDPG(env, agent, n_episodes, max_t, success_score, deque_len, print_every, brain_name)
        178             rewards = np.array(env_info.rewards).reshape(-1, 1)      # get the rewards
        179             dones = np.array(env_info.local_done).reshape(-1, 1)     # see if episode has finished
    --> 180             agent.step(states, actions, rewards, next_states, dones)
        181             states = next_states
        182             scores_per_episode.append(rewards)


    D:\ARQUIVOS PESSOAIS\GitHub\Reinforcement-Learning\DRL Nanodegree\Personal Exercises Solutios\4. Multi-Agent  Reinforcement Learning\p3_collab-compet\maddpg.py in step(self, state, action, reward, next_state, done)
        112            self._step >= self.set.UPDATE_EVERY):
        113             experiences = self.memory.sample()
    --> 114             self.learn(experiences, self.set.GAMMA)
        115             self._step = 0
        116


    D:\ARQUIVOS PESSOAIS\GitHub\Reinforcement-Learning\DRL Nanodegree\Personal Exercises Solutios\4. Multi-Agent  Reinforcement Learning\p3_collab-compet\maddpg.py in learn(self, experiences, gamma)
        176             critic_loss.backward()
        177             # Clip gradients
    --> 178             torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), self.set.CRITIC_GRADIENT_CLIP_VALUE)
        179             agent.critic_optimizer.step()
        180             # self.critic_lr_scheduler.step()


    D:\Miniconda\envs\mlagents\lib\site-packages\torch\nn\utils\clip_grad.py in clip_grad_norm_(parameters, max_norm, norm_type)
         31         for p in parameters:
         32             param_norm = p.grad.data.norm(norm_type)
    ---> 33             total_norm += param_norm.item() ** norm_type
         34         total_norm = total_norm ** (1. / norm_type)
         35     clip_coef = max_norm / (total_norm + 1e-6)


    KeyboardInterrupt:


### 5. Rewards


```python
import matplotlib.pyplot as plt

# Plot rewards
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
ax
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2d20ff41908>



### 6. Check the Trained Agent


```python
# load the weights from file
agent.actors_local[0].load_state_dict(torch.load('Actor1_checkpoint.pth'))
agent.actors_local[1].load_state_dict(torch.load('Actor2_checkpoint.pth'))
agent.critic_local.load_state_dict(torch.load('Critic_checkpoint.pth'))

# Use the trained agent for 1 episodes
for i in range(1):
    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    states = env_info.vector_observations            # get the current state
    score = 0
    while True:
        actions = agent.act(states)
        env_info = env.step(actions)[brain_name]                 # perform actions
        next_states = env_info.vector_observations               # get the next states
        rewards = np.array(env_info.rewards).reshape(-1, 1)      # get the rewards
        dones = np.array(env_info.local_done).reshape(-1, 1)     # see if episode has finished
        agent.step(states, actions, rewards, next_states, dones)
        states = next_states
        score += sum(rewards)
        if dones.any():
            break
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-13-ac7dca8bd663> in <module>
         13         env_info = env.step(actions)[brain_name]                 # perform actions
         14         next_states = env_info.vector_observations               # get the next states
    ---> 15         rewards = np.array(env_info.rewards).reshape(-1, 1)      # get the rewards
         16         dones = np.array(env_info.local_done).reshape(-1, 1)     # see if episode has finished
         17         agent.step(states, actions, rewards, next_states, dones)


    NameError: name 'np' is not defined


When finished, you can close the environment.


```python
env.close()
```

### 7. Future Improvements

Implement these methods to improve learning performance:

- Extract some improvements from the [Rainbow](https://arxiv.org/pdf/1710.02298.pdf) architecture such as:
    - Prioritized Experience
    - Multi-step learning
    - Noisy Nets
