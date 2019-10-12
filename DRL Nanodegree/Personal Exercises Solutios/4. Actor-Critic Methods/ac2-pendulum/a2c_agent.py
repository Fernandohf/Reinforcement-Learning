import numpy as np
import random
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Actor, Critic

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
N_STEP_BOOTSTRAP = 4         # boostrapping step size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts and learns from the environment."""

    def __init__(self, state_size, action_size, actor_hidden_size=(32),
                 critic_hidden_size=(32, 16), random_seed=42):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            actor_hidden_size (tuple): Dimension of hidden units for actor network
            critic_hidden_size (tuple): Dimension of hidden units for critic network
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network
        self.actor = Actor(state_size, action_size, actor_hidden_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        # Critic Network
        self.critic = Critic(state_size, action_size, critic_hidden_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device).unsqueeze(dim=0)

        # Forwards pass on policy
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        # Noise
        if add_noise:
            action += self.noise.sample()
        # Clipped action
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, states, actions, rewards, next_states, dones, gamma=GAMMA):
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

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor(next_states)
        discount = gamma ** np.arange(len(rewards))
        rewards = np.asarray(rewards) * discount.reshape(-1, 1)

        rewards = torch.tensor(rewards, dtype=float, device=device)
        Q_targets_next = self.critic(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute advantage - actor loss
        actor_loss = (Q_targets - Q_expected).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
