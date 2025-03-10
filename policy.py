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



class TransformerPolicy(nn.Module):
    """
    A small Transformer-based policy network for discrete actions.

    The design:
      - Each state (a vector of dimension `state_dim`) becomes one "token."
      - We add a learnable positional embedding for each token index.
      - We pass the sequence through a Transformer Encoder.
      - We take the hidden representation of the final token in the sequence
        to produce logits for the next action.
    """

    def __init__(
            self,
            state_dim: int,
            hidden_dim: int,
            num_heads: int,
            num_layers: int,
            num_actions: int,
            max_len: int = 64  # max sequence length (e.g. 12 for 12 layers)
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.max_len = max_len

        # 1) Project raw state_dim -> hidden_dim
        self.state_embedding = nn.Linear(state_dim, hidden_dim)

        # 2) Learnable positional embeddings: [max_len, hidden_dim]
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)

        # 3) Transformer Encoder
        #    Using nhead=num_heads, feedforward=4*hidden_dim is typical
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            activation="relu",
            batch_first=False  # PyTorch's nn.Transformer default is [seq, batch, dim]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 4) Output head -> produce logits for discrete actions
        self.action_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, states_seq: torch.Tensor) -> torch.Tensor:
        """
        states_seq: Tensor of shape [batch_size, seq_len, state_dim]
        Returns: logits over actions, shape [batch_size, num_actions]
        """
        B, T, _ = states_seq.shape

        # 1) Embed the states
        x = self.state_embedding(states_seq)  # (B, T, hidden_dim)

        # 2) Add positional embeddings
        #    We'll create positions = [0..T-1], repeated for each batch
        positions = torch.arange(T, device=states_seq.device).unsqueeze(0)  # shape [1, T]
        # Expand to match batch [B, T], then do embedding => shape [B, T, hidden_dim]
        pos_emb = self.pos_embedding(positions)  # (1, T, hidden_dim)
        pos_emb = pos_emb.expand(B, T, self.hidden_dim)

        x = x + pos_emb  # (B, T, hidden_dim)

        # 3) PyTorch Transformer expects [seq_len, batch_size, hidden_dim], so transpose
        x = x.transpose(0, 1)  # => (T, B, hidden_dim)

        # 4) Pass through transformer encoder
        x = self.transformer_encoder(x)  # still (T, B, hidden_dim)

        # 5) We'll take the *last token's* representation as "the state's" hidden
        #    for deciding the next action
        x_last = x[-1, :, :]  # shape (B, hidden_dim)

        # 6) Predict action logits
        logits = self.action_head(x_last)  # shape (B, num_actions)
        return logits

    def action_distribution(self, states_seq: torch.Tensor) -> ptd.Categorical:
        """Convert the Transformer outputs into a Categorical distribution."""
        logits = self.forward(states_seq)
        return ptd.Categorical(logits=logits)

    def act(self, states_seq: torch.Tensor, return_log_prob=False):
        """
        Sample an action from the policy.
        states_seq can be shape (seq_len, state_dim) or (batch_size, seq_len, state_dim).

        For single-environment usage (batch=1), you can feed shape (seq_len, state_dim).
        We'll handle the reshape internally.
        """
        if states_seq.ndim == 2:
            # => shape (seq_len, state_dim), add batch dimension => (1, seq_len, state_dim)
            states_seq = states_seq.unsqueeze(0)
        # else if states_seq.ndim == 3 => already [batch, seq_len, state_dim]

        dist = self.action_distribution(states_seq)
        action = dist.sample()  # shape [batch_size]

        if return_log_prob:
            log_prob = dist.log_prob(action)
            return action.cpu().numpy(), log_prob.detach().cpu().numpy()
        else:
            return action.cpu().numpy()
