import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from drlalgos.algos.maml.extra import scaling
from torch.distributions.multivariate_normal import MultivariateNormal

torch.manual_seed(0)
random.seed(0)


class PolicyNetwork(
    nn.Module
):  # create the PolicyNetwork class as a subclass for nn.module
    def __init__(self, env, gamma):
        super(PolicyNetwork, self).__init__()

        self.state_space = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # outputs the mean of
        self.l1 = nn.Linear(self.state_space, 64, bias=True)

        self.l2 = nn.Linear(64, 64, bias=True)
        self.l3 = nn.Linear(64, self.action_dim, bias=True)

        self.ones = torch.ones(1)
        self.sigma = nn.Linear(1, self.action_dim, bias=False)

        self.gamma = gamma

        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x, weights):
        if (
            not weights
        ):  # if no weights are provided, use the parameters stored into the model
            model = nn.Sequential(
                self.l1,
                nn.ReLU(),
                self.l2,
                nn.ReLU(),
                self.l3,
            )
            out1 = model(x)

            # variances

            variance = nn.Sequential(self.sigma, nn.Softplus())

            out2 = variance(self.ones)

            # out = torch.cat((out1, out2))

            return out1, out2
        else:  # "manual" forward pass if weights are provided
            x = F.linear(x, weights["l1.weight"], weights["l1.bias"])
            x = F.relu(x)
            x = F.linear(x, weights["l2.weight"], weights["l2.bias"])
            x = F.relu(x)
            out1 = F.linear(x, weights["l3.weight"], weights["l3.bias"])

            out2 = F.linear(self.ones, weights["sigma.weight"])
            out2 = F.softplus(out2)

            return out1, out2


class ValueNetwork(
    nn.Module
):  # create the Value Network for a subclass of Neural Networks module
    def __init__(self, env, gamma):
        super(ValueNetwork, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.l1 = nn.Linear(self.state_space, 64, bias=True)
        self.l2 = nn.Linear(64, 64, bias=True)
        self.l3 = nn.Linear(64, 1, bias=True)

        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        self.gamma = gamma

    def forward(self, x, weights):
        if not weights:
            model = nn.Sequential(
                self.l1,
                # nn.Dropout(p=0.6),
                nn.ReLU(),
                self.l2,
                # nn.Dropout(p=0.6),
                nn.ReLU(),
                self.l3,
            )
            out = model(x)

            # raise NotImplementedError
        else:
            x = F.linear(x, weights["l1.weight"], weights["l1.bias"])
            x = F.relu(x)
            x = F.linear(x, weights["l2.weight"], weights["l2.bias"])
            x = F.relu(x)
            out = F.linear(x, weights["l3.weight"], weights["l3.bias"])

        return out


class Agent:
    def __init__(self, env, gamma):
        self.policy_net = PolicyNetwork(env, gamma)
        self.value_net = ValueNetwork(env, gamma)
        self.gamma = gamma

    def select_action(self, state, weights=[]):
        out1, out2 = self.policy_net(state, weights)
        for i in out2:
            if i > 10:
                print(i)
                break
        m = MultivariateNormal(out1, torch.diag(1 / out2))
        action = m.sample()
        log_prob = m.log_prob(action)
        return (action.detach(), log_prob)

    def prob(self, states, actions, weights={}):
        out1, out2 = self.policy_net(states, weights)
        m = MultivariateNormal(out1, torch.diag(1 / out2))
        log_prob = m.log_prob(actions)
        entropy_bonus = m.entropy()
        return (log_prob, entropy_bonus)

    def sample_traj(
        self, env, K, H, weights_pol={}, weights_val={}
    ):  # env is the environment, K is the number of trajectories, N is the time horizon
        #  Collect Trajectories

        loss = torch.tensor(0.0)
        losses = []

        loss_V = torch.tensor(0.0)
        losses_V = []
        sum_reward = 0.0
        av_lengths = 0

        batch_actions = deque()
        batch_logprobs = deque()
        batch_V = deque()
        batch_G = deque()
        batch_states = deque()
        batch_advantage = deque()

        for trajectory_i in range(K):
            rewards = []
            values = []
            next_values = []

            env.reset()

            state = torch.tensor(env.reset())

            length = 0

            for t in range(H):  # for each time step in the episode
                length += 1
                action, log_prob = self.select_action(
                    state, weights_pol
                )  # the agent selects action based on current state

                value = self.value_net(
                    state, weights_val
                )  # value of state for other algorithms (ppb and ppo)

                next_state, reward, done, _ = env.step(
                    action
                )  # environment gives out next state and reward

                sum_reward += reward
                rewards.append(reward)

                batch_logprobs.append(log_prob)
                values.append(value)
                batch_actions.append(action)
                batch_states.append(state)

                state = torch.tensor(next_state)

                if done or (H == length):
                    next_values = values[1:]
                    next_values.append(self.value_net(state, weights_val).squeeze())
                    break

            # =============================================================================
            #             cum_steps += length
            #             cum_steps_epoch.append(cum_steps)
            #
            # =============================================================================
            av_lengths += length / K
            batch_V.extend(values)
            # Estimate Discounted Cumulative Rewards
            # First, compute reward to go G for each time period
            Gs = []  # cumulated discounted rewards
            R = torch.tensor(0.0)
            for rew in reversed(rewards):
                R = rew + self.gamma * R
                Gs.insert(0, R)
            batch_G.extend(Gs)
            # Compute Advantage for this trajectory
            Adv_t = []

            V = torch.stack(values).squeeze()
            Adv_t = torch.stack(Gs).squeeze() - V.detach()  # we compute the advantage

            batch_advantage.extend(Adv_t)
            # moving on to next trajectory
        # when all trajectories are done:
        av_reward = (
            sum_reward / K
        )  # average reward across all trajectories of this epoch

        # Scaling advantage
        A_k = torch.stack(
            list(batch_advantage)
        ).squeeze()  # convert list of tensors to 0 dimensional tensor
        A_k = scaling(
            A_k
        )  # we scale the advantage to normalize it by the mean and standard deviation

        T_batch_states = torch.stack(list(batch_states))
        # print(T_batch_states.size())
        T_batch_actions = torch.stack(list(batch_actions))
        # print(T_batch_actions.size())
        log_probs_old = (
            torch.stack(list(batch_logprobs)).squeeze().detach()
        )  # transform log_prob into tensor, but detach it from the computation graph so that no updates can go through it

        log_probs = torch.stack(list(batch_logprobs)).squeeze()

        # if algo == "ppo":
        #     log_probs, entropy = self.prob(T_batch_states, T_batch_actions)
        #     ratios = torch.exp(log_probs - log_probs_old)
        #     surr1 = ratios * A_k
        #     surr2 = torch.clamp(ratios, 1 - clip, 1 + clip) * A_k
        #     loss = (- torch.min(surr1, surr2)).mean() - 0.005*entropy.mean()

        # else: # pgb
        #     loss = -  (log_probs * A_k).mean() # computes the loss
        # # value function loss

        V = torch.stack(list(batch_V)).squeeze()
        G = torch.stack(list(batch_G)).squeeze()

        return (
            A_k,
            av_reward,
            log_probs,
            T_batch_states,
            T_batch_actions,
            V.squeeze(),
            G.detach(),
            av_lengths,
        )

        return (loss, loss_V)

    def get_dict_param_pol(self):
        param = {}

        for name, p in self.policy_net.named_parameters():
            param[name] = p

        return param

    def get_dict_param_val(self):
        param = {}

        for name, p in self.value_net.named_parameters():
            param[name] = p

        return param


class Log(object):
    def __init__(self, hyper_param):
        self.meta_iter = hyper_param.meta_iters
        self.meta_batch_size = hyper_param.meta_batch_size
        self.K = hyper_param.K
        self.H = hyper_param.H
        self.num_adapt_steps = hyper_param.num_adapt_steps
        self.meta_loss = deque([])
        self.meta_loss_V = deque([])
        self.val_returns = deque([])
        self.number_of_iter = 500

    def meta_append(self, val_return, meta_loss, meta_loss_V):
        self.val_returns.append(val_return)
        self.meta_loss.append(meta_loss)
        self.meta_loss_V.append(meta_loss_V)

    def append(self, av_reward, loss, loss_V):
        self.av_rewards.append(av_reward)
        self.steps.append(self.N * self.T)
        self.S += self.N * self.T
        self.cum_steps.append(self.S)
        self.losses.append(loss)
        self.losses_V.append(loss_V)
