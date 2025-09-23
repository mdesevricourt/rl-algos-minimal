import random

import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

torch.manual_seed(0)
random.seed(0)


class PolicyNetwork(
    nn.Module
):  # create the PolicyNetwork class as a subclass for nn.module
    """Policy Network.
    Args:
        env: OpenAI environment.
        gamma: Discount factor.
    Returns:
        action: action sampled from the policy network.
        log_prob: log probability of the action.
    """

    def __init__(self, env, gamma):
        super(PolicyNetwork, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.l1 = nn.Linear(self.state_space, 128, bias=True)

        self.l2 = nn.Linear(128, 128, bias=True)
        self.l3 = nn.Linear(128, 128, bias=True)
        self.l4 = nn.Linear(128, 128, bias=True)
        self.l5 = nn.Linear(128, self.action_dim, bias=True)

        self.ones = torch.ones(1)
        self.sigma = nn.Linear(1, self.action_dim, bias=False)

        self.gamma = gamma

        # Episode policy and reward history
        # self.policy_history = Variable(torch.Tensor())

        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        model = nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l3,
            # nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l4,
            # nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l5,
        )
        out1 = model(x)

        # variances

        variance = nn.Sequential(self.sigma, nn.Softplus())

        out2 = variance(self.ones)

        # out = torch.cat((out1, out2))

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

    def __init__(self, env, gamma):
        self.policy_net = PolicyNetwork(env, gamma)
        # self.value_net = ValueNetwork()

    def select_action(self, state):
        out1, out2 = self.policy_net(state)
        for i in out2:
            if i <= 0:
                print(out2)
                break
        m = MultivariateNormal(out1, torch.diag(out2))
        action = m.sample()
        log_prob = m.log_prob(action)
        return (action, log_prob)
