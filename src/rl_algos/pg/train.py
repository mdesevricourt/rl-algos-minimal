"""
Created on Thu Oct  7 12:18:38 2021

@author: Maxime Cugnon de Sévricourt

This module implements the training loop for the PPO (Proximal Policy Optimization) algorithm.
It collects trajectories, computes returns, and updates the policy using collected experience.
"""

import os
import random
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import pandas as pd
import pybullet_envs_gymnasium
import torch
import torch.optim as optim

from rl_algos.pg.agent import Agent
from rl_algos.pg.extra import Trajectory, Transition

# print(sys.argv)

os.chdir(os.path.dirname(__file__))
torch.manual_seed(0)
random.seed(0)


# %% TrainConfig Class


@dataclass
class TrainConfig:
    """Configuration for training parameters."""

    epoch: int = 50  # number of epochs to train
    N: int = 5  # number of trajectories per epoch
    T: int = 999  # number of steps per trajectory
    env: str = "AntBulletEnv-v0"  # environment name
    algo: str = "ppo"  # algorithm name
    lr: float = 0.01
    gamma: float = 0.99  # discount factor
    target_update: int = 1  # number of epochs between two print statements

    def __str__(self) -> str:
        return f"TrainConfig(epoch={self.epoch}, N={self.N}, T={self.T}, env='{self.env}', algo='{self.algo}', lr={self.lr}, gamma={self.gamma}, target_update={self.target_update})"


# %% Policy


class OnPolicyBuffer(object):
    """
    Buffer for storing statistics and rewards during on-policy training.

    Attributes:
        env: The environment instance.
        cfg: Training configuration parameters.
    """

    def __init__(self, env, cfg: TrainConfig):
        """
        Initialize the buffer.

        Args:
            env: The environment instance.
            cfg: Training configuration parameters.
        """
        self.env = env
        self.epoch = cfg.epoch
        self.gamma = cfg.gamma
        self.T = cfg.T
        self.N = cfg.N
        self.reward_history: deque = deque([])
        self.S_max = cfg.epoch * cfg.N * cfg.T  # total number of steps
        self.av_rewards: deque = deque([])  # list of average rewards
        self.steps: deque = deque([])
        self.cum_steps: deque = deque([])
        self.S = 0
        self.algo = cfg.algo

    def append(self, av_reward: float):
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


def main(cfg: TrainConfig) -> OnPolicyBuffer:
    """
    Main training loop for PPO.

    Args:
        cfg: Training configuration parameters.
    Returns:
        buffer: OnPolicyBuffer containing training statistics.
    """
    print("Training Configuration:")
    print(cfg)

    N = cfg.N  # number of trajectories per epoch
    T = cfg.T  # number of steps per trajectory

    env = gym.make(cfg.env)
    agent = Agent(env, cfg.gamma)
    optimizer = optim.SGD(agent.policy_net.parameters(), lr=cfg.lr)
    buffer = OnPolicyBuffer(env, cfg)

    for i_epoch in range(cfg.epoch):

        #  Collect Trajectories

        logpiG_ls: deque[torch.Tensor] = deque([])
        sum_reward = torch.tensor(0.0)

        for _ in range(N):
            trajectory = Trajectory(cfg.gamma)
            env.reset()

            state, info = env.reset()
            state = torch.tensor(state)

            length = torch.tensor(0, dtype=torch.float32)
            for t in range(T):  # for each time step in the episode
                length += 1
                action, log_prob = agent.select_action(
                    state
                )  # the agent selects action based on current state
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

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

        av_reward: torch.Tensor = sum_reward / torch.tensor(
            N
        )  # average reward across all trajectories of this epoch

        sum_logpiG = torch.stack(list(logpiG_ls)).sum()

        # Improve Policy
        optimizer.zero_grad()
        loss: torch.Tensor = -sum_logpiG / N
        loss.backward()
        optimizer.step()

        # Log Statistics
        buffer.append(float(av_reward))

        if i_epoch % cfg.target_update == 0:
            print(
                "Epoch {}\t, Average sum of rewards: {:.2f}\t, Last loss: {:.2f}".format(
                    i_epoch, av_reward, loss
                )
            )

    env.close()
    return buffer


# %%
def parse_args() -> TrainConfig:
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
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()

    buffer = main(cfg)

    rolling_mean = pd.Series(buffer.av_rewards).rolling(10).mean()
