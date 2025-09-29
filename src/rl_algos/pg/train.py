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
import matplotlib.pyplot as plt
import pandas as pd
import pybullet_envs_gymnasium
import torch
import torch.optim as optim

from rl_algos.pg.agent import Agent
from rl_algos.pg.extra import Episode, Trajectory, Transition
from rl_algos.pg.trainconfig import TrainConfig
from rl_algos.pg.trainer import PG, PPO, Algo, PGBaseline

# print(sys.argv)

os.chdir(os.path.dirname(__file__))
torch.manual_seed(0)
random.seed(0)


# %% TrainConfig Class


# %%
def make_trainer(agent: Agent, cfg: TrainConfig):
    if cfg.algo == Algo.PG:
        return PG(agent, cfg)
    if cfg.algo == Algo.VPG:
        return PGBaseline(agent, cfg)
    if cfg.algo == Algo.PPO:
        return PPO(agent, cfg)
    raise ValueError(cfg.algo)


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
    agent = Agent(env)
    buffer = OnPolicyBuffer(env, cfg)
    trainer = make_trainer(agent, cfg)

    for i_epoch in range(cfg.epoch):

        #  Collect Trajectories

        episode = Episode()

        for _ in range(N):
            trajectory = Trajectory(cfg.gamma)

            state, info = env.reset()
            state = torch.tensor(state)

            length = torch.tensor(0, dtype=torch.float32)
            for t in range(T):  # for each time step in the episode
                length += 1
                action, log_prob = agent.select_action(
                    state
                )  # the agent selects action based on current state
                action_np = action.detach().cpu().numpy()
                next_state, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated

                transition = Transition(state, action, next_state, reward, log_prob)
                trajectory.push(transition)

                state = torch.tensor(next_state)
                if done:

                    print("Trajectory stopped at time ", t)
                    break

            # Estimate Returns
            trajectory.end_trajectory()  # compute discounted sum of rewards etc
            episode.append(trajectory)

        # Improve Policy
        trainer.update(episode)

        # Log Statistics
        buffer.append(float(episode.average_reward()))

        if i_epoch % cfg.target_update == 0:
            print(
                f"Epoch {i_epoch},\tAverage sum of rewards: {episode.average_reward():.2f},\tPolicy Loss: {trainer.last_policy_loss:.3f}, Value Loss: {trainer.last_value_loss:.3f}\t"
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
        default="ppo",
        type=str,
        help="Name of algorithm. It should be one of [pg, pgb, ppo]",
    )
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()

    buffer = main(cfg)

    rolling_mean = pd.Series(buffer.av_rewards).rolling(10).mean()
    # add title with config
    plt.title(f"Training Curve: {cfg.env} with {cfg.algo.upper()}")
    plt.plot(buffer.cum_steps, buffer.av_rewards, label="Average Reward")
    plt.plot(buffer.cum_steps, rolling_mean, label="Rolling Mean (10)")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.show()
