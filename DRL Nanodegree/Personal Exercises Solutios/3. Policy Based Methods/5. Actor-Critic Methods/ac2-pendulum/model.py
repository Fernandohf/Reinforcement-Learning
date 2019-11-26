"""Network Models used to solve the problem """
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCActor(nn.Module):
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
        super(FCActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Model
        self.layers = nn.ModuleList()
        layers_size = [state_size] + list(hidden_size) + [action_size]
        for i in range(len(layers_size) - 1):
            self.layers.append(nn.Linear(layers_size[i], layers_size[i + 1]))
        # Gaussian distribution std
        self.std = nn.Parameter(torch.zeros(action_size))

    def forward(self, x):
        """Build an actor (policy) network that maps states -> actions."""
        # Passthrough all the layers
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        # Final layer with tanh activation for the joints
        mean = 2 * torch.tanh(self.layers[-1](x))
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        action = dist.rsample()
        log_prob = dist.log_prob(action).mean(-1).unsqueeze(-1)
        return action, log_prob


class LSTMActor(nn.Module):
    """Actor (policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=128, seed=42):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_size (int): Number of nodes in hidden layers
            seed (int): Random seed
        """
        super(LSTMActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Model
        self.lstm_layer = nn.LSTM(input_size=state_size, hidden_size=hidden_size,
                                  num_layers=2, batch_first=True, dropout=.1)

        # Gaussian distribution std
        self.std = nn.Parameter(torch.zeros(action_size))

    def forward(self, x):
        """Build an actor (policy) network that maps states -> actions."""
        # Passthrough the layers
        x, (h, c) = self.lstm_layer(x)
        # Final layer with tanh activation for the joints
        mean = 2 * torch.tanh(x[:, -1, -1].view(-1, 1))
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        action = dist.rsample()
        log_prob = dist.log_prob(action).mean(-1).unsqueeze(-1)
        return action, log_prob


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

        layers_size = [state_size + action_size] + list(hidden_size) + [1]
        for i in range(len(layers_size) - 1):
            self.layers.append(nn.Linear(layers_size[i], layers_size[i + 1]))

    def forward(self, state, action):
        """Build an critic (value function) network that maps states ,actions -> value function."""
        # Join states and actions
        x = torch.cat((state, action), dim=1)
        # Forward pass
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        # No activation  function on final layer
        return self.layers[-1](x)
