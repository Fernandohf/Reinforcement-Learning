"""Network Models used to solve the problem """
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Actor (policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=(512, 256), seed=42):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_size (tuple): Number of nodes in hidden layers
            seed (int): Random seed
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Model
        self.layers = nn.ModuleList()

        for i, s in enumerate(hidden_size):
            if i == 0:
                self.layers.append(nn.Linear(state_size, hidden_size[i]))
            elif i == len(hidden_size) - 1:
                self.layers.append(nn.Linear(hidden_size[i], action_size))
            else:
                self.layers.append(nn.Linear(hidden_size[i - 1], hidden_size[i]))

    def forward(self, x):
        """Build an actor (policy) network that maps states -> actions."""
        # Passthrough all the layers
        for l in self.layers[:-1]:
            x = torch.relu(self.l(x))
        # Final layer with tanh activation for the joints
        return torch.tanh(self.layers[-1](x))


class Critic(nn.Module):
    """Critic (value function) Model."""

    def __init__(self, state_size, action_size, hidden_size=(512, 256), seed=42):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_size (tuple): Number of nodes in hidden layers
            seed (int): Random seed
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Model
        self.layers = nn.ModuleList()

        for i, s in enumerate(hidden_size):
            if i == 0:
                self.layers.append(nn.Linear(state_size + action_size, s))
            elif i == len(hidden_size) - 1:
                self.layers.append(nn.Linear(s, 1))
            else:
                self.layers.append(nn.Linear(hidden_size[i - 1], s))

    def forward(self, state, action):
        """Build an critic (value function) network that maps states ,actions -> value function."""
        x = torch.cat((state, action), dim=1)
        for l in self.layers:
            x = F.relu(self.l(x))
        return x
