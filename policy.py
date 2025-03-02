# policy.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as ptd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class PolicyNetPPO(nn.Module):
    """
    A simple multi-layer perceptron that outputs logits over discrete actions
    (the bit-width choices). We can integrate with PPO by computing log_probs.
    """
    def __init__(self, state_dim, hidden_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x):
        """Compute raw logits from state."""
        return self.net(x)

    def action_distribution(self, states):
        """
        states: [batch_size, state_dim]
        returns a Categorical distribution over the discrete action space
        """
        logits = self.forward(states)
        return ptd.Categorical(logits=logits)

    def act(self, state, return_log_prob=False):
        """
        state: np.array, shape [state_dim], a single observation (or batch).
        We'll handle the single state => batch dimension inside.
        """
        if state.ndim == 1:
            state = state[None, ...]  # make [1, state_dim]

        state_t = torch.from_numpy(state).float().to(device)
        dist = self.action_distribution(state_t)

        action_t = dist.sample()  # shape [batch_size], but often batch=1
        if return_log_prob:
            logprob_t = dist.log_prob(action_t)
            return action_t.numpy(), logprob_t.detach().cpu().numpy()
        else:
            return action_t.cpu().numpy()
