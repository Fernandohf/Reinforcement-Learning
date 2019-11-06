# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

import torch
import numpy as np
from model import Actor, Critic
from torch.optim import Adam
import torch.nn.functional as F
from utilities import ReplayBuffer, OUNoise, GaussianNoise
from types import SimpleNamespace

# Hyperparameters
PARAMETERS = {
    'BUFFER_SIZE': int(1e6),          # Replay buffer size
    'BATCH_SIZE': 128,                # Minibatch size
    'GAMMA': 1.0,                     # Discount factor
    'TAU': 1e-3,                      # Soft update of target parameters
    'UPDATE_EVERY': 10,               # Wait for more experiences before update

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

    'OU_THETA': .8,                   # OU noise theta parameter
    'OU_SIGMA': .6,                   # OU noise sigma parameters


    'SEED': 42,                       # Random seed
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
DEFAULT_SETTINGS = SimpleNamespace(**PARAMETERS)


class MADDPG:
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

        # MADDPG Actors/Critic - Local and Target Networks - Decentralized
        self.actors_local = []
        self.actors_target = []
        self.actors_optimizer = []
        for i in range(self.set.N_AGENTS):
            self.actors_local.append(Actor(self.set.STATE_SIZE, self.set.ACTION_SIZE,
                                           self.set.SEED).to(self.set.DEVICE))
            self.actors_target.append(Actor(self.set.STATE_SIZE, self.set.ACTION_SIZE,
                                            self.set.SEED).to(self.set.DEVICE))
            self.actors_optimizer.append(Adam(self.actors_local[i].parameters(),
                                              lr=self.set.ACTOR_LR,
                                              weight_decay=self.set.ACTOR_WEIGHT_DECAY))

        # MADDPG Critic - Local and Target Networks - Centralized
        self.critic_local = Critic(self.set.STATE_SIZE * self.set.N_AGENTS,
                                   self.set.ACTION_SIZE * self.set.N_AGENTS,
                                   self.set.N_AGENTS,
                                   self.set.SEED).to(self.set.DEVICE)
        self.critic_target = Critic(self.set.STATE_SIZE * self.set.N_AGENTS,
                                    self.set.ACTION_SIZE * self.set.N_AGENTS,
                                    self.set.N_AGENTS,
                                    self.set.SEED).to(self.set.DEVICE)
        self.critic_optimizer = Adam(self.critic_local.parameters(),
                                     lr=self.set.CRITIC_LR,
                                     weight_decay=self.set.CRITIC_WEIGHT_DECAY)

        # Noise process
        if self.set.NOISE_TYPE == 'ou':
            self.noise = OUNoise(self.set.ACTION_SIZE * self.set.N_AGENTS, self.set.SEED,
                                 theta=self.set.OU_THETA, sigma=self.set.OU_SIGMA)
        else:
            self.noise = GaussianNoise(self.set.ACTION_SIZE * self.set.N_AGENTS, self.set.SEED,
                                       sigma=self.set.N_SIGMA)
        # Replay buffer
        self.memory = ReplayBuffer(self.set.BUFFER_SIZE, self.set.BATCH_SIZE,
                                   self.set.SEED, self.set.DEVICE)
        # Steps counter
        self._step = 0

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
        for i in range(self.set.N_AGENTS):
            state = torch.from_numpy(states[i][np.newaxis]).float().to(self.set.DEVICE)
            self.actors_local[i].eval()
            with torch.no_grad():
                actions.append(self.actors_local[i](state).cpu().data.numpy())
            self.actors_local[i].train()
        actions = np.vstack(actions)

        # Add noise
        if add_noise:
            actions += self.noise.sample().reshape(actions.shape)

        # Clip actions values
        return np.clip(actions, -1, 1)

    def reset(self):
        """
        Auxiliary function to reset OU noise
        """
        self.noise.reset()

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

        # ---------------------------- Update Critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models for each agent
        actions_next = []
        # For each agent
        for curr_agent in range(self.set.N_AGENTS):  # BS, AGENT, SIZE
            actions_next = [self.actors_target[a](next_states[:, a, :]) if a == curr_agent else
                            self.actors_target[a](next_states[:, a, :]).detach()         # Detach other agent
                            for a in range(self.set.N_AGENTS)]
            actions_next = torch.stack(actions_next)

            Q_targets_next = self.critic_target(next_states.view(-1, self.set.N_AGENTS * self.set.STATE_SIZE),
                                                actions_next.view(-1, self.set.N_AGENTS * self.set.ACTION_SIZE))
            # Compute Q targets for current states (y_i)
            Q_targets = (rewards.view(-1, self.set.N_AGENTS)[:, curr_agent] +
                         (gamma * Q_targets_next[:, curr_agent] * (1 - dones.view(-1, self.set.N_AGENTS)[:, curr_agent])))
            # Compute critic loss
            Q_expected = self.critic_local(states.view(-1, self.set.N_AGENTS * self.set.STATE_SIZE),
                                           actions.view(-1, self.set.N_AGENTS * self.set.ACTION_SIZE))[:, curr_agent]
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.set.CRITIC_GRADIENT_CLIP_VALUE)
            self.critic_optimizer.step()
            # self.critic_lr_scheduler.step()

            # ---------------------------- Update Actor ---------------------------- #
            # Compute actor loss
            actions_pred = [self.actors_target[a](states[:, a, :]) if a == curr_agent else
                            self.actors_target[a](states[:, a, :]).detach()         # Detach other agent
                            for a in range(self.set.N_AGENTS)]
            actions_pred = torch.stack(actions_pred, dim=1)
            actor_loss = -self.critic_local(states.view(-1, self.set.N_AGENTS * self.set.STATE_SIZE),
                                            actions_pred.view(-1, self.set.N_AGENTS * self.set.ACTION_SIZE)).mean(dim=0)[curr_agent]
            agent_optimizer = self.actors_optimizer[curr_agent]
            agent_optimizer.zero_grad()
            actor_loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.actors_local[curr_agent].parameters(),
                                           self.set.ACTOR_GRADIENT_CLIP_VALUE)
            agent_optimizer.step()
            # self.actor_lr_scheduler.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local, self.critic_target, self.set.TAU)
            self.soft_update(self.actors_local[curr_agent],
                             self.actors_target[curr_agent], self.set.TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ * θ_local + (1 - τ) * θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_models(self, name):
        """
        Saves the current state of the the actors and the critic.
        """
        # Actors
        for i, a in enumerate(self.actors_local):
            torch.save(a.state_dict(), f'Actor{i + 1}_{name}.pth')
        # Critic
        torch.save(self.critic_local.state_dict(), f'Critic_{name}.pth')
        torch.save(self._parameters, f'Hyperparameters_{name}.pth')
