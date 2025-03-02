# baseline_network.py
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaselineNetwork(nn.Module):
    """
    Baseline MLP that predicts the value (critic) given a state.
    Output dimension = 1.
    """
    def __init__(self, state_dim, hidden_dim, lr=1e-3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        """
        x: [batch_size, state_dim]
        returns: [batch_size], the predicted values
        """
        return self.net(x).squeeze(-1)

    def predict(self, states):
        """
        states: numpy array or torch tensor, shape [batch_size, state_dim]
        returns: numpy array of shape [batch_size]
        """
        if not isinstance(states, torch.Tensor):
            states = torch.from_numpy(states).float()
        states = states.to(device)

        with torch.no_grad():
            values = self.forward(states)

        return values.cpu().numpy()

    def update(self, states, targets):
        """
        states: np array [batch_size, state_dim]
        targets: np array [batch_size], the returns or V-target
        Minimizes MSE between predicted V(s) and targets
        """
        self.optimizer.zero_grad()

        states_t = torch.from_numpy(states).float().to(device)
        targets_t = torch.from_numpy(targets).float().to(device)
        values_t = self.forward(states_t)

        loss = torch.mean((values_t - targets_t) ** 2)
        loss.backward()
        self.optimizer.step()

        return loss.item()
