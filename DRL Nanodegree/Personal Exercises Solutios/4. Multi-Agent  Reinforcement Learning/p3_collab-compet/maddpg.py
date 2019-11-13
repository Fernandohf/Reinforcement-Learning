# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

import torch
import numpy as np
import torch.nn.functional as F
from utilities import ReplayBuffer, OUNoise, GaussianNoise
from ddpg import DDPGAgent
from types import SimpleNamespace


# Hyperparameters
PARAMETERS = {
    'BUFFER_SIZE': int(1e6),          # Replay buffer size
    'BATCH_SIZE': 128,                # Minibatch size
    'GAMMA': 1.0,                     # Discount factor
    'TAU': 5e-3,                      # Soft update of target parameters
    'UPDATE_EVERY': 1,               # Wait for more experiences before update

    'N_AGENTS': 2,                    # Total number of agents
    'STATE_SIZE': 24,                 # Size of the state for each agent
    'ACTION_SIZE': 2,                 # Size of actions for each agent

    'ACTOR_LR': 1e-3,                 # Learning rate of the actor
    'ACTOR_WEIGHT_DECAY': 0.0,        # Actor L2 weight decay
    'ACTOR_GRADIENT_CLIP_VALUE': 10,  # Max gradient modulus for clipping

    'CRITIC_LR': 1e-3,                # Learning rate of the critic
    'CRITIC_WEIGHT_DECAY': 0.0,       # Critic L2 weight decay
    'CRITIC_GRADIENT_CLIP_VALUE': 2,  # Max gradient modulus for clipping


    'NOISE_TYPE': 'normal',           # Type of noise used: 'normal' or 'ou'
    'N_SIGMA': 3,                     # Normal noise sigma parameters

    'OU_THETA': .01,                   # OU noise theta parameter
    'OU_SIGMA': .01,                  # OU noise sigma parameters


    'SEED': 42,                       # Random seed
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
DEFAULT_SETTINGS = SimpleNamespace(**PARAMETERS)


class MADDPG():
    """
    Implements a Multi Agent Deep Deterministic Policy Gradient.
    It uses a centralized training approach to train the critic and decentralized for the actors.

    Parametes
    ----------
    settings: Settings object
        An object with all the hyperparameters for the model.
        If not prodived the default settings will be used (see above).

    """
    def __init__(self, parameters=None):
        super(MADDPG, self).__init__()

        # Parameters
        if parameters is None:
            self.set = DEFAULT_SETTINGS
            self._parameters = PARAMETERS
        else:
            self.set = SimpleNamespace(**parameters)
            self._parameters = parameters

        # Create agents
        self.agents = [DDPGAgent(self._parameters) for i in range(self.set.N_AGENTS)]

        # Noise process
        if self.set.NOISE_TYPE == 'ou':
            self.noise = OUNoise(self.set.ACTION_SIZE * self.set.N_AGENTS, self.set.SEED,
                                 theta=self.set.OU_THETA, sigma=self.set.OU_SIGMA)
        else:
            self.noise = GaussianNoise(self.set.ACTION_SIZE * self.set.N_AGENTS, self.set.SEED,
                                       self.set.N_MEAN, self.set.N_SIGMA, self.set.N_EPS_BETA,
                                       self.set.N_EPS_INIT, eps_min=self.set.N_EPS_MIN)
        # Replay buffer
        self.memory = ReplayBuffer(self.set.BUFFER_SIZE, self.set.BATCH_SIZE,
                                   self.set.SEED, self.set.DEVICE)
        # Steps counter
        self._step = 0

    def reset(self):
        self.noise.reset()

    def step(self, state, action, reward, next_state, done):
        """
        Save experience in replay memory, and use random sample from buffer to learn.

        Parameters
        ----------
        state: np.ndarray
            States from the environment
        action: np.ndarray
            Actions from the environment
        reward: np.ndarray
            Rewards from the environment
        next_states: np.ndarray
            Next states from the environment given actions
        done: np.ndarray
            Weather or not the environment has reached the end
        """
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        # self.update_every = self._step_count
        self._step += 1
        # Learn, if enough samples are available in memory
        if (len(self.memory) > self.set.BATCH_SIZE and
           self._step >= self.set.UPDATE_EVERY):
            experiences = self.memory.sample()
            self.learn(experiences, self.set.GAMMA)
            self._step = 0

    def act(self, states, add_noise=True):
        """
        Returns actions for given state as per current policy.

        Parameters
        ----------
        states: np.ndarray
            States from the environment with shape (N_AGENTS, STATE_SIZE)
        """
        # Actions for each agent
        actions = []
        for i, agent in enumerate(self.agents):
            state = torch.from_numpy(states[i, np.newaxis]).float().to(self.set.DEVICE)
            agent.actor_local.eval()
            with torch.no_grad():
                actions.append(agent.actor_local(state).cpu().data.numpy())
            agent.actor_local.train()
        actions = np.vstack(actions)

        # Add noise
        if add_noise:
            actions += self.noise.sample().reshape(actions.shape)

        return actions

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Parameters
        ----------
            experiences: Tuple[torch.Tensor]
                Tuple with (s, a, r, s', done) and first dimension equal to BATCH_SIZE
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # For each agent
        for agent_n, agent in enumerate(self.agents):  # BS, AGENT, SIZE
            actions_next = [self.agents[a].actor_target(next_states[:, a, :]) if a == agent_n else
                            self.agents[a].actor_target(next_states[:, a, :]).detach()         # Detach other agent
                            for a in range(self.set.N_AGENTS)]
            actions_next = torch.stack(actions_next)

            Q_targets_next = agent.critic_target(next_states.view(-1, self.set.N_AGENTS * self.set.STATE_SIZE),
                                                 actions_next.view(-1, self.set.N_AGENTS * self.set.ACTION_SIZE))
            # Compute Q targets for current states (y_i)
            Q_targets = (rewards[:, agent_n] +
                         (gamma * Q_targets_next * (1 - dones[:, agent_n])))
            # Compute critic loss
            Q_expected = agent.critic_local(states.view(-1, self.set.N_AGENTS * self.set.STATE_SIZE),
                                            actions.view(-1, self.set.N_AGENTS * self.set.ACTION_SIZE))
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), self.set.CRITIC_GRADIENT_CLIP_VALUE)
            agent.critic_optimizer.step()
            # self.critic_lr_scheduler.step()

            # ---------------------------- Update Actor ---------------------------- #
            # Compute actor loss
            actions_pred = [agent.actor_target(states[:, a, :]) if a == agent_n else
                            agent.actor_target(states[:, a, :]).detach()         # Detach other agent
                            for a in range(self.set.N_AGENTS)]
            actions_pred = torch.stack(actions_pred, dim=1)
            actor_loss = -agent.critic_local(states.view(-1, self.set.N_AGENTS * self.set.STATE_SIZE),
                                             actions_pred.view(-1, self.set.N_AGENTS * self.set.ACTION_SIZE)).mean()
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(agent.actor_local.parameters(),
                                           self.set.ACTOR_GRADIENT_CLIP_VALUE)
            agent.actor_optimizer.step()
            # self.actor_lr_scheduler.step()

            # ----------------------- update target networks ----------------------- #
            agent.soft_update(agent.critic_local, agent.critic_target, self.set.TAU)
            agent.soft_update(agent.actor_local, agent.actor_target, self.set.TAU)

    def save_models(self, name):
        """
        Saves the current state of the the actors and the critic.
        """
        # Save agents status
        for i, a in enumerate(self.agents):
            # Actors
            torch.save(a.actor_local.state_dict(), f'Actor{i + 1}_{name}.pth')
            # Critic
            torch.save(a.critic_local.state_dict(), f'Critic{i + 1}_{name}.pth')
            torch.save(self._parameters, f'Hyperparameters_{name}.pth')
