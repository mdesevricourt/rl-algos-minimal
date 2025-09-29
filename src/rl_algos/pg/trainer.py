import select
from enum import Enum
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import optim

from rl_algos.pg.agent import Agent
from rl_algos.pg.extra import Episode, Trajectory
from rl_algos.pg.trainconfig import TrainConfig


class Algo(str, Enum):
    PG = "pg"  # REINFORCE / Vanilla Policy Gradient
    VPG = "pgb"  # PG with baseline
    PPO = "ppo"


class Trainer(Protocol):
    last_policy_loss: float = 0.0
    last_value_loss: float = 0.0

    def update(self, episode: Episode) -> None: ...


class PG(Trainer):
    def __init__(self, agent: Agent, cfg: TrainConfig) -> None:
        self.agent = agent
        self.optimizer = optim.Adam(agent.policy_net.parameters(), lr=cfg.lr)
        self.cfg = cfg

    def update(self, episode: Episode) -> None:
        self.optimizer.zero_grad()
        loss = -episode.sum_logpiG / len(episode.trajectories)
        loss.backward()
        self.optimizer.step()
        self.last_policy_loss = loss.item()


class PGBaseline(Trainer):
    def __init__(self, agent: Agent, cfg: TrainConfig) -> None:
        self.agent = agent
        self.optimizer_policy = optim.Adam(agent.policy_net.parameters(), lr=cfg.lr)
        self.optimizer_value = optim.Adam(agent.value_net.parameters(), lr=cfg.lr)
        self.cfg = cfg

    def update(self, episode: Episode):
        """
        Expects:
          episode.trajectories: list[Trajectory]
          each Trajectory has:
            - transitions: list[Transition(state, action, next_state, reward, log_prob)]
            - returns(): torch.Tensor of shape [T] (discounted MC returns)
        """
        # ---- accumulate losses over all steps in the episode batch
        value_loss = torch.zeros((), dtype=torch.float32)
        policy_loss = torch.zeros((), dtype=torch.float32)
        total_steps = 0

        for traj in episode.trajectories:
            # stack per-trajectory tensors
            states = torch.stack(list(traj.states))  # [T, state_dim]
            logprobs = torch.stack(list(traj.log_probs))  # [T]
            returns = traj.G  # [T]

            # value predictions
            values = self.agent.value_net(states).squeeze(-1)  # [T]

            # ---- value loss (MSE)
            value_loss = value_loss + F.mse_loss(values, returns, reduction="sum")

            # ---- advantages for policy update
            adv = returns - values.detach()  # [T]
            # (optional) normalize for variance reduction
            if adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # ---- policy loss (REINFORCE with baseline)
            policy_loss = policy_loss - (logprobs * adv).sum()

            total_steps += len(traj.states)

        # Avoid divide-by-zero
        total_steps = max(total_steps, 1)
        value_loss = value_loss / total_steps
        policy_loss = policy_loss / total_steps

        # ---- value step
        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()

        # ---- policy step
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()
        self.last_value_loss = value_loss.item()
        self.last_policy_loss = policy_loss.item()


class PPO(Trainer):
    def __init__(self, agent: Agent, cfg: TrainConfig) -> None:
        self.agent = agent
        self.optimizer_policy = optim.Adam(agent.policy_net.parameters(), lr=cfg.lr)
        self.optimizer_value = optim.Adam(agent.value_net.parameters(), lr=cfg.lr)
        self.cfg = cfg
        self.eps_clip = 0.2  # PPO clipping parameter
        self.K_epochs = 4  # PPO update epochs
        self.entropy_coef = 0.01  # entropy bonus coefficient
        self.value_coef = 0.5  # value loss coefficient
        self.max_grad_norm = 0.5  # gradient clipping

    def update(self, episode: Episode) -> None:
        states, actions, old_logprobs, returns = [], [], [], []
        for traj in episode.trajectories:
            states.append(torch.stack(list(traj.states)))
            actions.append(torch.stack(list(traj.actions)))
            old_logprobs.append(torch.stack(list(traj.log_probs)))
            returns.append(traj.G)
        states = torch.cat(states)  # [T_total, ...]
        actions = torch.cat(actions)  # [T_total, ...]
        old_logprobs = torch.cat(old_logprobs).detach()  # [T_total]
        returns = torch.cat(returns).detach()  # [T_total]
        batch_size = states.size(0)

        mini_batch_size = max(
            32, batch_size // (8 * self.K_epochs)
        )  # at least 32, at most 8 minibatches per epoch

        # 2) Compute values and advantages
        with torch.no_grad():
            values = self.agent.value_net(states).squeeze(-1)
            deltas = returns - values
            advantages = deltas  # replace with GAE-λ if you have V(s_{t+1})
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3) K PPO epochs with shuffled minibatches
        idx = torch.randperm(states.size(0))
        for _ in range(self.K_epochs):
            idx = idx[torch.randperm(idx.numel())]  # reshuffle each epoch
            for mb in idx.split(mini_batch_size):
                mb_states = states[mb]
                mb_actions = actions[mb]
                mb_old_logprobs = old_logprobs[mb]
                mb_returns = returns[mb]
                mb_advantages = advantages[mb]

                dist = self.agent.policy_dist(mb_states)
                new_logprobs = dist.log_prob(mb_actions)
                if new_logprobs.dim() > 1:
                    new_logprobs = new_logprobs.sum(-1)
                entropy = dist.entropy()
                if entropy.dim() > 1:
                    entropy = entropy.sum(-1)
                entropy = entropy.mean()

                ratios = (new_logprobs - mb_old_logprobs).exp()
                surr1 = ratios * mb_advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * mb_advantages
                )
                policy_loss = (
                    -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                )

                values_pred = self.agent.value_net(mb_states).squeeze(-1)
                # Optional value clipping:
                mb_values_old = values[mb]
                values_clipped = mb_values_old + (values_pred - mb_values_old).clamp(
                    -self.eps_clip, self.eps_clip
                )
                vloss_unclipped = F.mse_loss(values_pred, mb_returns)
                vloss_clipped = F.mse_loss(values_clipped, mb_returns)
                value_loss = torch.max(vloss_unclipped, vloss_clipped)

                # Separate optimizers, separate graphs
                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.agent.policy_net.parameters(), self.max_grad_norm
                )
                self.optimizer_policy.step()

                self.optimizer_value.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.agent.value_net.parameters(), self.max_grad_norm
                )
                self.optimizer_value.step()

        self.last_policy_loss = policy_loss.item()
        self.last_value_loss = value_loss.item()
