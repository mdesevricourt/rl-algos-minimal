# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 16:03:38 2021

@author: Maxime Cugnon de Sévricourt
"""
from collections import deque

import torch


class Trajectory:
    def __init__(self, gamma):
        self.gamma = gamma
        self.rewards = deque([])
        self.sum_reward = torch.tensor(0.0)
        self.log_probs = []

        # self.states= deque([])
        # self.actions = deque([])

    def push(self, Transition):  # add a transition to the trajectory
        # self.states.append(Transition.state)
        # self.actions.append(Transition.action)
        # self.cum_rewards.append(sum(self.rewards))

        self.rewards.append(Transition.reward)
        self.log_probs.append(Transition.log_prob)
        self.sum_reward += Transition.reward

    def end_trajectory(
        self,
    ):  # compute sum of reward, discounted cumulative reward G and the dot product of G and logpi

        gamma = self.gamma

        R = 0.0
        G1 = deque([])
        rewards = torch.tensor(self.rewards)
        scaled_rewards = (rewards - rewards.mean()) / (
            rewards.std() + torch.finfo(rewards.dtype).eps
        )

        for r in reversed(scaled_rewards):  # calculate discounted rewards
            R = r + gamma * R
            G1.appendleft(R)

        # G1 = deque([])
        # rewards = torch.tensor(self.rewards)
        # for i in range(len(self.rewards)):
        #     dis_factors = torch.tensor([gamma**j for j in range(len(self.rewards) - i)])
        #     G1.append(torch.dot(dis_factors, rewards[i:]))

        # scale the rewards
        G2 = torch.tensor(G1)
        self.G = G2

        self.logpiG = torch.stack(self.log_probs) * self.G

        self.sum_logpiG = sum(self.logpiG)


class Episode:
    def __init__(self):
        self.sum_rewards = torch.tensor(0.0)
        self.undisc_sum_rewards = torch.tensor(0.0)
        self.sum_logpiG = torch.tensor(0.0)

    def append(self, trajectory):
        self.sum_rewards += sum(trajectory.G)
        print(trajectory.undis_reward)
        self.undisc_sum_rewards += sum(trajectory.undis_reward)
        self.sum_logpiG += sum(trajectory.logpiG)
