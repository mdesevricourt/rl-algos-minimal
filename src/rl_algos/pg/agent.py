import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

torch.manual_seed(0)
random.seed(0)


class PolicyNetwork(nn.Module):
    """Policy Network.

    Args:
        env: OpenAI environment.
        gamma: Discount factor.
        num_layers: number of hidden layers.
        layer_size: size of each hidden layer.
    Returns:
        action: action sampled from the policy network.
        log_prob: log probability of the action.
    """

    def __init__(self, env, gamma: float, num_layers: int = 3, layer_size: int = 128):
        """Initialize the PolicyNetwork.
        The network builds `num_layers` hidden linear layers of size `layer_size`.
        Args:
            env: OpenAI environment.
            gamma: Discount factor.
            num_layers: number of hidden layers.
            layer_size: neurons per hidden layer.
        """
        super(PolicyNetwork, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # Build hidden layers dynamically
        hidden_layers = []
        # first hidden layer from state to layer_size
        hidden_layers.append(nn.Linear(self.state_space, layer_size, bias=True))
        # remaining hidden layers (if any)
        for _ in range(max(0, num_layers - 1)):
            hidden_layers.append(nn.Linear(layer_size, layer_size, bias=True))

        self.hidden_layers = nn.ModuleList(hidden_layers)
        # final output layer maps last hidden to action_dim
        self.output = nn.Linear(layer_size, self.action_dim, bias=True)

        self.ones = torch.ones(1)
        self.sigma = nn.Linear(1, self.action_dim, bias=False)
        # variance module (ensure positive output)
        self.variance = nn.Sequential(self.sigma, nn.Softplus())

        self.gamma = gamma

        self.reward_episode: list[float] = []
        self.reward_history: list[float] = []
        self.loss_history: list[float] = []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: state tensor of shape (state_dim,) or (batch, state_dim)

        Returns:
            (mean, scale) where `mean` has shape (..., action_dim) and `scale` is a positive tensor
            of shape (action_dim,) representing per-dimension scale (shared across batch).
        """
        out = x
        for linear in self.hidden_layers:
            out = linear(out)
            out = F.dropout(out, p=0.6, training=self.training)
            out = F.relu(out)

        out1 = self.output(out)

        out2 = self.variance(self.ones)

        return out1, out2


class ValueNetwork:
    def __init__(self):
        pass
        # raise NotImplementedError


class Agent:
    """PPO Agent.
    Args:
        env: OpenAI environment.
        gamma: Discount factor.
    Returns:
        action: action sampled from the policy network.
        log_prob: log probability of the action.
    """

    def __init__(self, env, gamma: float):
        """Initialize the Agent.
        Args:
            env: OpenAI environment.
            gamma: Discount factor.
        """
        self.policy_net = PolicyNetwork(env, gamma)
        # self.value_net = ValueNetwork()

    def select_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select action according to the policy network.
        Args:
            state: current state
        Returns:
            action: action sampled from the policy network.
            log_prob: log probability of the action.
        """
        out1, out2 = self.policy_net(state)
        for i in out2:
            if i <= 0:
                print(out2)
                break
        m = MultivariateNormal(out1, torch.diag(out2))
        action = m.sample()
        log_prob = m.log_prob(action)
        return (action, log_prob)
