# Multi Agent Reinforcement Learning

This project implements the MADDPG algorithm to solve the [Unity Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

## The Tennis Environment

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

- **Observation Space**: The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket for each agent.
- **Action Space**: Two continuous actions per agent are available, corresponding to movement toward (or away from) the net, and jumping.
- **Solution**: In order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

## The Algorithm

The algorithm used to solve the environment is called [Multi Agent Deep Deterministic Policy Gradients](https://arxiv.org/pdf/1706.02275.pdf). It is a multi agent version of the [DDPG](https://arxiv.org/pdf/1509.02971.pdf), which is suitable for continuous control scenarios (such as this Tennis task). Similarly to its single agent counterpart, **MADDPG** is an off-policy, actor-critic algorithm that uses the concept of local and target networks. Therefore, it uses a **Replay Buffer** to reduce consecutive correlations biases, as well as **Soft Updates** to stabilize training.

The main difference is that it uses the framework of **Centralized Training with Decentralized Execution**. Briefly, it consists of using the maximum amount of information during training, but restricting this information during testing time. In the **MADDPG**, this implies that the critic networks for each agents has access to other agents actions and states during training. While in execution time, only actors are present, therefore, they can only see their own agents states and actions.

## Methodology

The descriptions of the choices taken during the resolution of this project, such as network architecture and hyperparameters.

### Networks Architecture

Actor network receives an its own state `(24)` and outputs two continuous actions `(2)`. Two batch normalization layers were used on the actor.

![Actor Network](https://github.com/Fernandohf/Reinforcement-Learning/blob/master/DRL%20Nanodegree/Personal%20Exercises%20Solutios/4.%20Multi-Agent%20%20Reinforcement%20Learning/p3_collab-compet/results/Actor_NN.png?raw=true "Actor network representation")

Critic network receives all states`(48)` and actions `(4)`, and outputs a single value used to evaluate the actor `(1)`. In this case, it is being used a critic for each agent. Only one batch normalization layers was used on the critic.

![Critic Network](https://github.com/Fernandohf/Reinforcement-Learning/blob/master/DRL%20Nanodegree/Personal%20Exercises%20Solutios/4.%20Multi-Agent%20%20Reinforcement%20Learning/p3_collab-compet/results/Critic_NN.png?raw=true "Critic network representation")

### Hyperparameters

The algorithm was executed for **2000** episodes. As soon as the conditions for solving the environment were met,the networks weights were saved, but the algorithm continued to see if further improvements could be achieved. The algorithm results were recorded using the [Weight and Biases](https://app.wandb.ai/fernandohf/maddpg) tool. It records, hyperparameters, losses, network weights, system information and  more(check the link for more information). All the hyperparameters with their description are available on the file `config.yml` and also shown below.

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

The model achieved the solved condition (average score above **0.5** on last 100 episodes) at episode **929** and continued improving until episode **1003** with average score of **0.928**. From there, the performance dropped a little and then started increasing to achieve the highest during this training session, **1.539** at episode **1169**. From that point forwards, the performance only dropped significantly even when below the solved threshold and did not recover.

The main training graphs are plotted below and are also available on the [W&B project site](https://app.wandb.ai/fernandohf/maddpg).

### Plots

Average Score on last 100 Episodes

![Average Rewards plot](https://github.com/Fernandohf/Reinforcement-Learning/blob/master/DRL%20Nanodegree/Personal%20Exercises%20Solutios/4.%20Multi-Agent%20%20Reinforcement%20Learning/p3_collab-compet/results/avg.png?raw=true "Average Rewards Plot")

Smoothed Scores

![Smoothed Scores plot](https://github.com/Fernandohf/Reinforcement-Learning/blob/master/DRL%20Nanodegree/Personal%20Exercises%20Solutios/4.%20Multi-Agent%20%20Reinforcement%20Learning/p3_collab-compet/results/score.png?raw=true "Smoothed Scores Plot")

### Trained Agents

The gif below is a recording of the trained agent just after solving the task.

![Trained agents performance](https://github.com/Fernandohf/Reinforcement-Learning/blob/master/DRL%20Nanodegree/Personal%20Exercises%20Solutios/4.%20Multi-Agent%20%20Reinforcement%20Learning/p3_collab-compet/results/trained_agents.gif?raw=true "Trained Agent performance")

## Future Improvements

Extract some improvements from the [Rainbow](https://arxiv.org/pdf/1710.02298.pdf) architecture such as:

- **Prioritized Experience**
- **Multi-step learning**
- **Noisy Nets**
