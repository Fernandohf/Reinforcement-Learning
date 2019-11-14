import random
import torch
import numpy as np
from copy import copy
from collections import namedtuple, deque
from itertools import islice
import wandb


class GaussianNoise:
    """Normal noise added"""

    def __init__(self, size, seed, mu=0., sigma=.2,
                 eps_beta=.01, eps_init=1, eps_min=.001):
        """Initialize parameters and noise process."""
        self.mu = mu
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self._eps_init = eps_init
        self._eps = self._eps_init
        self._eps_min = eps_min
        self._eps_beta = eps_beta
        self.reset()

    def reset(self):
        """Reset the internal epsilon decay status."""
        # Epsilon
        self._eps = self._eps_init
        self._eps_step = 0

    def sample(self):
        """Update internal state and return it as a noise sample."""
        # Update epsilon
        self._eps = max([np.exp(-self._eps_beta * self._eps_step), self._eps_min])
        self._eps_step += 1

        return np.random.normal(loc=self.mu,
                                scale=self.sigma,
                                size=self.size) * self._eps


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=.15, sigma=.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e
                                            is not None], axis=0)).float().to(self.device)
        actions = torch.from_numpy(np.stack([e.action for e in experiences if e
                                             is not None], axis=0)).float().to(self.device)
        rewards = torch.from_numpy(np.stack([e.reward for e in experiences if e
                                             is not None], axis=0)).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e
                                                 is not None], axis=0)).float().to(self.device)
        dones = torch.from_numpy(np.stack([e.done for e in experiences if e
                                           is not None], axis=0).astype(np.uint8)).float().to(self.device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def train_MADDPG(env, agent, n_episodes, max_t, success_score, brain_name,
                 deque_len=100, print_every=100, use_wandb=True):
    """
    Training method for Multi Agent Deep Deterministic Policy Gradients.

    Parameters
    ----------
    n_episodes: int
        Maximum number of training episodes
    max_t: int
        Maximum number of timesteps per episode
    success_score: float
        Average score to consider the task solved.
    """
    scores_deque = deque(maxlen=deque_len)
    sum_scores_per_agent = []
    scores = []
    solved = False
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]   # reset the environment
        states = env_info.vector_observations               # get the current states
        scores_per_episode = []
        agent.reset()
        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]                 # perform actions
            next_states = env_info.vector_observations               # get the next states
            rewards = np.array(env_info.rewards).reshape(-1, 1)      # get the rewards
            dones = np.array(env_info.local_done).reshape(-1, 1)     # see if episode has finished
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            scores_per_episode.append(rewards)
            if dones.any():
                break
        # Scores
        scores_per_episode = np.concatenate(scores_per_episode, axis=1)              # Scores along the episode
        sum_scores_per_agent.append(scores_per_episode.sum(axis=1).reshape(-1, 1))   # Sum of scores per agent
        max_score = np.max(scores_per_episode.sum(axis=1).reshape(-1, 1))            # Max agent sum of scores
        scores_deque.append(max_score)
        scores.append(max_score)
#         import pdb; pdb.set_trace()

        mean_deque_score = np.mean(scores_deque)
        print('\rEpisode {:3d}/{}\tAverage Score (last {:3d}): {:.4f}\t Last score: {:.4f}'.format(i_episode,
                                                                                                   n_episodes,
                                                                                                   min(i_episode, deque_len),
                                                                                                   mean_deque_score,
                                                                                                   max_score), end="")
        # WandB logging
        if use_wandb:
            wandb.log({"Episode": i_episode,
                       "Score": max_score,
                       "Average Score": mean_deque_score})
        if i_episode % print_every == 0:
            reversed_scores = copy(scores)
            reversed_scores.reverse()
            mean_last_n = np.mean(list(islice(reversed_scores, 0, print_every)))
            agent.save_models("checkpoint")
            print('\rEpisode {:3d}/{}\tAverage Score (last {:3d}): {:.4f}\t Last score: {:.4f}'.format(i_episode,
                                                                                                       n_episodes,
                                                                                                       print_every,
                                                                                                       mean_last_n,
                                                                                                       max_score))
        if mean_deque_score >= success_score and not solved:
            agent.save_models("solved")
            print('\rSolved on Episode {:3d}/{}\tAverage Score (last {:3d}): {:.4f}'.format(i_episode,
                                                                                            n_episodes,
                                                                                            min(i_episode, deque_len),
                                                                                            mean_deque_score))
            wandb.log({"Episode Solved": i_episode})
            solved = True

    return scores
