"""
Created on Thu Oct  7 12:18:38 2021

@author: Maxime Cugnon de Sévricourt

This module implements the training loop for the PPO (Proximal Policy Optimization) algorithm.
It collects trajectories, computes returns, and updates the policy using collected experience.
"""

import os
import random
from collections import deque, namedtuple

import gym
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from drlalgos.algos.ppo.agent import Agent
from drlalgos.algos.ppo.extra import Trajectory

# print(sys.argv)

os.chdir(os.path.dirname(__file__))
torch.manual_seed(0)
random.seed(0)

# %% Hyperparameters

learning_rate = 0.01
gamma = 0.99
TARGET_UPDATE = 1  # number of epochs between two print statements


# %% Policy

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "log_prob")
)


class OnPolicyBuffer(object):
    """
    Buffer for storing statistics and rewards during on-policy training.

    Attributes:
        env: The environment instance.
        epoch: Number of epochs to train.
        gamma: Discount factor.
        T: Number of steps per trajectory.
        N: Number of trajectories per epoch.
        reward_history: Deque storing reward history.
        av_rewards: Deque storing average rewards per epoch.
        steps: Deque storing steps per epoch.
        cum_steps: Deque storing cumulative steps.
        S: Total number of steps.
        algo: Name of the algorithm.
    """

    def __init__(self, env, args, gamma, N, T):
        """
        Initialize the buffer.

        Args:
            env: The environment instance.
            args: Arguments containing epoch and algo.
            gamma: Discount factor.
            N: Number of trajectories per epoch.
            T: Number of steps per trajectory.
        """
        self.env = env
        self.epoch = args.epoch
        self.gamma = gamma
        self.T = T
        self.N = N
        self.reward_history = deque([])
        self.S = args.epoch * N * T  # total number of steps
        self.av_rewards = deque([])  # list of average rewards
        self.steps = deque([])
        self.cum_steps = deque([])
        self.S = 0
        self.algo = args.algo

    def append(self, av_reward):
        """
        Append average reward and update step counters.

        Args:
            av_reward: Average reward for the epoch.
        """
        self.av_rewards.append(av_reward)
        self.steps.append(self.N * self.T)
        self.S += self.N * self.T
        self.cum_steps.append(self.S)


# %%


def main(args):
    """
    Main training loop for PPO.

    Args:
        args: Parsed command-line arguments.

    Returns:
        buffer: OnPolicyBuffer containing training statistics.
    """
    N = 5  # number of trajectories per epoch
    T = 999  # number of steps per trajectory

    env = gym.make(args.env)
    agent = Agent(env, gamma)
    optimizer = optim.SGD(agent.policy_net.parameters(), lr=learning_rate)

    buffer = OnPolicyBuffer(env, args, gamma, N, T)

    for i_epoch in range(args.epoch):

        #  Collect Trajectories

        logpiG_ls = deque([])
        sum_reward = torch.tensor(0.0)

        for trajectory_i in range(N):
            trajectory = Trajectory(gamma)
            env.reset()

            state = torch.tensor(env.reset())

            length = 0
            for t in range(T):  # for each time step in the episode
                length += 1
                action, log_prob = agent.select_action(
                    state
                )  # the agent selects action based on current state
                next_state, reward, done, _ = env.step(action)

                transition = Transition(state, action, next_state, reward, log_prob)
                trajectory.push(transition)

                state = torch.tensor(next_state)
                if done:
                    # episode_durations.append(t + 1)
                    # plot_durations()
                    print("Trajectory stopped at time ", t)
                    break

            # Estimate Returns
            trajectory.end_trajectory()  # compute discounted sum of rewards etc
            logpiG_ls.append(trajectory.sum_logpiG / length)

            sum_reward += trajectory.sum_reward

        av_reward = (
            sum_reward / N
        )  # average reward across all trajectories of this epoch

        sum_logpiG = sum(logpiG_ls)  # compute

        # Improve Policy
        optimizer.zero_grad()
        loss = -sum_logpiG / N
        loss.backward()
        optimizer.step()

        # Log Statistics
        buffer.append(av_reward)

        if i_epoch % TARGET_UPDATE == 0:
            print(
                "Epoch {}\t, Average sum of rewards: {:.2f}\t, Last loss: {:.2f}".format(
                    i_epoch, av_reward, loss
                )
            )

    env.close()
    return buffer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="AntBulletEnv-v0")
    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument(
        "--algo",
        required=True,
        type=str,
        help="Name of algorithm. It should be one of [pg, pgb, ppo]",
    )

    args = parser.parse_args()

    buffer = main(args)

    rolling_mean = pd.Series(buffer.av_rewards).rolling(10).mean()
