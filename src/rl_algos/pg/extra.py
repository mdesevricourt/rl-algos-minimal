"""
Created on Fri Oct  8 16:03:38 2021

@author: Maxime Cugnon de Sévricourt
"""

from collections import deque, namedtuple

import torch

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "log_prob")
)


class Trajectory:
    """Trajectory class to store states, actions, rewards, log probabilities, and compute returns.
    Args:
        gamma: Discount factor.
    """

    def __init__(self, gamma: float):
        """
        Initialize the Trajectory. It stores rewards, log probabilities, and computes discounted returns.
        Args:
            gamma: Discount factor.
        """
        self.gamma = gamma
        self.rewards: deque[float] = deque([])
        self.sum_reward: torch.Tensor = torch.tensor(0.0)
        self.log_probs: deque[torch.Tensor] = deque([])

        self.states: deque[torch.Tensor] = deque([])
        self.actions: deque[torch.Tensor] = deque([])
        self.cum_rewards: deque[float] = deque([])

    def push(self, transition: Transition) -> None:
        """Append a new transition to the trajectory.
        Args:
            transition: A named tuple containing state, action, reward, log_prob.
        """
        self.states.append(transition.state)
        self.actions.append(transition.action)
        self.cum_rewards.append(sum(self.rewards))

        self.rewards.append(transition.reward)
        self.log_probs.append(transition.log_prob)
        self.sum_reward += transition.reward

    def end_trajectory(
        self,
    ) -> None:
        """Compute discounted rewards and related quantities at the end of the trajectory.
        Args:
            None

        Returns:
        None
        """

        gamma = self.gamma

        R = 0.0
        G1: deque[float] = deque([])

        if len(self.rewards) == 0:
            # nothing to do
            self.G = torch.tensor([])
            self.logpiG = torch.tensor([])
            self.sum_logpiG = torch.tensor(0.0)
            return

        rewards = torch.tensor(list(self.rewards), dtype=torch.get_default_dtype())
        scaled_rewards = (rewards - rewards.mean()) / (
            rewards.std() + torch.finfo(rewards.dtype).eps
        )

        for r in reversed(scaled_rewards):  # calculate discounted rewards
            R = r + gamma * R
            G1.appendleft(float(R))

        # G1 = deque([])
        # rewards = torch.tensor(self.rewards)
        # for i in range(len(self.rewards)):
        #     dis_factors = torch.tensor([gamma**j for j in range(len(self.rewards) - i)])
        #     G1.append(torch.dot(dis_factors, rewards[i:]))

        # scale the rewards
        G2 = torch.tensor(list(G1), dtype=rewards.dtype)
        self.G = G2

        log_probs = torch.tensor(list(self.log_probs), dtype=rewards.dtype)
        self.logpiG = log_probs * self.G

        self.sum_logpiG = torch.sum(self.logpiG)


class Episode:
    def __init__(self) -> None:
        self.sum_rewards = torch.tensor(0.0)
        self.undisc_sum_rewards = torch.tensor(0.0)
        self.sum_logpiG = torch.tensor(0.0)

    def append(self, trajectory: Trajectory) -> None:
        # accumulate discounted-sum (G), undiscounted rewards and sum_logpiG
        if hasattr(trajectory, "G"):
            self.sum_rewards += torch.sum(trajectory.G)
        # undiscounted sum: trajectory.rewards is a deque of numbers
        self.undisc_sum_rewards += torch.tensor(
            sum(trajectory.rewards), dtype=torch.get_default_dtype()
        )
        # use precomputed sum_logpiG if available
        if hasattr(trajectory, "sum_logpiG"):
            self.sum_logpiG += trajectory.sum_logpiG
