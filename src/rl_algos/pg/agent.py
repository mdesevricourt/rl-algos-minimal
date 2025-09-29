import random

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal


class PolicyNetwork(nn.Module):
    """
    Discrete-action policy network.
    - forward(x) returns logits of shape [B, action_dim] (or [action_dim] for 1D input)
    - Sampling/log-probs are handled by your Agent.
    """

    def __init__(
        self,
        env: gym.Env,
        num_layers: int = 3,
        layer_size: int = 128,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()

        obs_shape = getattr(env.observation_space, "shape", None)
        assert obs_shape and len(obs_shape) == 1, "Expected flat observation space"
        self.state_dim = int(obs_shape[0])
        assert env.action_space and len(env.action_space.shape) == 1
        self.action_dim = int(env.action_space.shape[0])

        self.dropout_p = float(dropout_p)

        layers: list[nn.Module] = []
        in_dim = self.state_dim

        if num_layers <= 0:
            # no hidden layers: direct linear to logits
            self.hidden_layers = nn.ModuleList([])
            self.output = nn.Linear(in_dim, self.action_dim)
        else:
            # first hidden layer
            layers.append(nn.Linear(in_dim, layer_size))
            layers.append(nn.ReLU())
            if self.dropout_p > 0:
                layers.append(nn.Dropout(self.dropout_p))

            # remaining hidden layers (num_layers - 1)
            for _ in range(max(0, num_layers - 1)):
                layers.append(nn.Linear(layer_size, layer_size))
                layers.append(nn.ReLU())
                if self.dropout_p > 0:
                    layers.append(nn.Dropout(self.dropout_p))

            self.hidden_layers = nn.ModuleList(layers)
            self.output = nn.Linear(layer_size, self.action_dim)

        self.ones = torch.ones(1)
        self.sigma = nn.Linear(1, self.action_dim, bias=False)
        self.variance = nn.Sequential(self.sigma, nn.Softplus())

        # (Optional) a simple, robust init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [state_dim] or [B, state_dim]
        Returns:
            logits: [action_dim] or [B, action_dim]
        """
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        out = x
        for layer in self.hidden_layers:
            out = layer(out)
        logits = self.output(out)
        variance = self.variance(self.ones)

        return logits, variance


class ValueNetwork(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        num_layers: int = 3,
        layer_size: int = 128,
    ) -> None:
        super().__init__()

        state_dim = env.observation_space.shape[0]

        layers = []
        in_dim = state_dim

        # hidden layers
        for _ in range(num_layers - 1):  # last layer is output
            layers.append(nn.Linear(in_dim, layer_size))
            layers.append(nn.ReLU())
            in_dim = layer_size

        # output layer → scalar value
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Agent:
    """Agent.
    Args:
        env: OpenAI environment.
        gamma: Discount factor.
    Returns:
        action: action sampled from the policy network.
        log_prob: log probability of the action.
    """

    def __init__(self, env: gym.Env) -> None:
        """Initialize the Agent.
        Args:
            env: OpenAI environment.
        """
        self.policy_net = PolicyNetwork(env)
        self.value_net = ValueNetwork(env)

    def select_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select action according to the policy network.
        Args:
            state: current state
        Returns:
            action: action sampled from the policy network.
            log_prob: log probability of the action.
        """
        dist = self.policy_dist(state.unsqueeze(0))
        action: torch.Tensor = dist.sample()
        log_prob: torch.Tensor = dist.log_prob(action)

        return (action, log_prob)

    def policy_dist(self, states: torch.Tensor) -> MultivariateNormal:
        """Get the action distribution for given states.
        Args:
            states: [B, state_dim]
        Returns:
            dist: a torch.distributions object representing the action distribution.
        """
        mean, var = self.policy_net(states)
        if mean.dim() == 2 and mean.size(0) == 1:
            mean = mean.squeeze(0)
        if var.dim() == 2 and var.size(0) == 1:
            var = var.squeeze(0)

        cov = torch.diag_embed(var)
        dist = MultivariateNormal(mean, cov)
        return dist
