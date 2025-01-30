# policy.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """
    A simple feed-forward policy network for discrete actions.
    """
    def __init__(self, state_dim=5, hidden_dim=32, num_actions=4):
        """
        :param state_dim: dimensionality of the environment state
        :param hidden_dim: size of hidden layer
        :param num_actions: number of discrete actions (e.g., 4 possible bit-widths)
        """
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        """
        :param x: shape [batch, state_dim]
        :return: shape [batch, num_actions] (logits)
        """
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits
