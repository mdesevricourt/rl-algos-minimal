import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from agent import Agent, Log
from maml_env import HalfCheetahDirecBulletEnv

from drlalgos.algos.maml.extra import (
    hyperparameters,
    inner_loop,
    plot_grad_flow,
    plot_learning,
    scaling,
    update_function,
)

# print("Number of processors: ", mp.cpu_count())
# pool = mp.Pool(mp.cpu_count())

# %%


class Tasks:
    def __init__(self, *task_configs):
        self.tasks = [i for i in task_configs]

    def sample_tasks(self, batch_size):
        return random.choices(self.tasks, k=batch_size)


class Param(object):
    def __init__(self):
        self.meta_iters = 10
        self.H = 200
        self.meta_batch_size = 10
        self.K = 10  # number of trajectories sampled
        self.num_adapt_steps = 1
        (
            self.learning_rate,
            self.learning_rate_value_net,
            self.gamma,
            self.TARGET_UPDATE,
            self.UPDATE_V,
            self.nb_updates_per_epoch,
            self.clip,
        ) = hyperparameters()  # load hyperparameters


# %%


def main(args):

    Path = os.getcwd() + "\\" + "Saved models"
    isExist = os.path.exists(Path)
    if not isExist:
        os.makedirs(Path)
        print("Created directory to store models")

    Path_log = os.getcwd() + "\\" + "Log"
    isExist = os.path.exists(Path_log)
    if not isExist:
        os.makedirs(Path_log)
        print("Created directory to store results")

    hyper_param = Param()

    hyper_param.meta_iters = args.meta_iteration
    hyper_param.H = args.horizon
    hyper_param.meta_batch_size = args.meta_batch_size
    hyper_param.num_adapt_steps = args.num_adapt_steps

    tasks = Tasks(("Forward", True), ("Backward", False))

    log = Log(hyper_param)

    # initialize the agent
    env = HalfCheetahDirecBulletEnv()
    agent = Agent(env, hyper_param.gamma)

    optimizer = torch.optim.Adam(
        agent.policy_net.parameters(), lr=hyper_param.learning_rate
    )
    optimizer_V = torch.optim.Adam(
        agent.value_net.parameters(), lr=hyper_param.learning_rate
    )
    val_return_outer = []
    last_update = 0  # track the last time the best model was updated
    # Outer loop
    for meta_iter in range(hyper_param.meta_iters):
        meta_losses = []
        meta_losses_V = []
        val_return_inner = []
        av_length = 0

        for task_config in tasks.sample_tasks(hyper_param.meta_batch_size):
            adapted_loss, adapted_loss_V, av_reward, av_length2 = inner_loop(
                agent, task_config, hyper_param
            )

            meta_losses.append(adapted_loss)
            meta_losses_V.append(adapted_loss_V)
            val_return_inner.append(av_reward)
            av_length += av_length2 / hyper_param.meta_batch_size

        # Meta Optimization of the policy function
        optimizer.zero_grad()
        # agent.policy_net.zero_grad() # set gradient to zero
        meta_loss = sum(meta_losses)  # compute meta loss by adding all the losses
        meta_loss.backward()  # compute the gradient
        optimizer.step()
        optimizer.zero_grad()

        # Meta Optimization of the value network
        optimizer_V.zero_grad()
        # agent.policy_net.zero_grad() # set gradient to zero
        meta_loss_V = sum(meta_losses_V)  # compute meta loss by adding all the losses
        meta_loss_V.backward()  # compute the gradient
        optimizer_V.step()
        optimizer_V.zero_grad()

        # Log statistics
        val_return = np.mean(val_return_inner)
        log.meta_append(
            val_return, meta_loss.clone().detach(), meta_loss_V.clone().detach()
        )
        val_return_outer.append(val_return)
        last_update += 1
        if meta_iter % hyper_param.TARGET_UPDATE == 0:
            print(
                "Iter {}\t, Average validation rewards: {:.2f},\t Average length: {:.2f},\t Loss: {:.4f},\t VF loss: {:.2f}".format(
                    meta_iter, val_return, av_length, meta_loss, meta_loss_V
                )
            )

        # save parameters if model has the highest validation return so far
        if val_return >= max(val_return_outer):
            last_update = 0
            print(
                "Saving model at iter: {},\t with average validation rewards: {:.2f}\t".format(
                    meta_iter, val_return
                )
            )
            filename = "MAML" + str(hyper_param.num_adapt_steps)
            f = Path + "\\" + filename
            torch.save(
                {
                    "meta-iteration": meta_iter,
                    "policy_state_dict": agent.policy_net.state_dict(),
                    "value_state_dict": agent.value_net.state_dict(),
                    "Validation returns": val_return,
                    "Number of adaptatio steps": hyper_param.num_adapt_steps,
                },
                f,
            )

            os.chdir(Path_log)
            logname = "log_MAML" + str(hyper_param.num_adapt_steps) + ".obj"
            with open(logname, "wb") as filehandler:
                pickle.dump(log, filehandler)

        if last_update >= 25:
            print("No update to the models for 50 iterations, interrupting training")
            log.number_of_iter = meta_iter
            break

    os.chdir(Path_log)
    logname = "log_MAML" + str(hyper_param.num_adapt_steps) + ".obj"
    with open(logname, "wb") as filehandler:
        pickle.dump(log, filehandler)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--meta_iteration", default=500, type=int)
    parser.add_argument("--meta_batch_size", default=10, type=int)
    parser.add_argument("--horizon", "-H", default=200, type=int)
    parser.add_argument("--num_adapt_steps", default=1, type=int)
    parser.add_argument("--test", default=False, type=bool)
    args = parser.parse_args()

    main(args)
