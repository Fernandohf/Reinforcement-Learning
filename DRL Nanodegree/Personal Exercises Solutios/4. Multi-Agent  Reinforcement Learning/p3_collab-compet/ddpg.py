import numpy as np
import random
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from torch.optim import Adam

from model import Actor, Critic
from utilities import OUNoise, GaussianNoise, ReplayBuffer


class DDPGAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, parameters):
        """
        Initialize an Deep Deterministic Policy Gradient Agent.

        Parameters
        ----------
        parameters: dict
            All parameters to create the agent.
        """
        # Parameters
        self.set = SimpleNamespace(**parameters)
        self._parameters = parameters

        # Hyper parameters
        self.seed = random.seed(self.set.SEED)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(self.set.STATE_SIZE, self.set.ACTION_SIZE, self.set.SEED).to(self.set.DEVICE)
        self.actor_target = Actor(self.set.STATE_SIZE, self.set.ACTION_SIZE, self.set.SEED).to(self.set.DEVICE)
        self.actor_optimizer = Adam(self.actor_local.parameters(),
                                    lr=self.set.ACTOR_LR,
                                    weight_decay=self.set.ACTOR_WEIGHT_DECAY)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.set.STATE_SIZE, self.set.ACTION_SIZE,
                                   self.set.N_AGENTS, self.set.SEED).to(self.set.DEVICE)
        self.critic_target = Critic(self.set.STATE_SIZE, self.set.ACTION_SIZE,
                                    self.set.N_AGENTS, self.set.SEED).to(self.set.DEVICE)
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

        # Replay memory
        self.memory = ReplayBuffer(self.set.ACTION_SIZE, self.set.BUFFER_SIZE, self.set.BATCH_SIZE, self.set.SEED)

        self._step_count = 0

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        self._step_count += 1

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.set.BATCH_SIZE and self._step_count > self.set.UPDATE_EVERY:
            self._step_count = 0
            experiences = self.memory.sample()
            self.learn(experiences, self.set.GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.set.DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        # Add noise
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.setTAU)
        self.soft_update(self.actor_local, self.actor_target, self.set.TAU)

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
