# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 16:03:38 2021

@author: Maxime Cugnon de Sévricourt
"""
import torch 
from collections import namedtuple, deque
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from maml_env import HalfCheetahDirecBulletEnv


def inner_loop(agent, task_config, hyper_param):
    # keep track of validation returns
     task_name, env_args = task_config[0], task_config[1:]
     env = HalfCheetahDirecBulletEnv(*env_args)
     
     param_pol = agent.get_dict_param_pol()
     param_val = agent.get_dict_param_val()
     
     for step_adap in range(hyper_param.num_adapt_steps):
         # Run agent in environment, returns advantage, av_reward, log probabities, states visited, actions taken, Value function and cumulative reward
         A_k, av_reward, log_probs, T_batch_states, T_batch_actions, V, G, av_lengths_traj = agent.sample_traj(env, hyper_param.K, hyper_param.H, param_pol, param_val) 
         #av_length += av_lengths_traj/hyper_param.meta_batch_size
 
         # Adaptation
         # set gradients to zero if already computed in previous loop
         for name, p in param_pol.items():
             if p.grad == None:
                 continue
             else: 
                 p.grad.data.zero_()
                 
         
         for name, p in param_val.items():
             if p.grad == None:
                 continue
             else: 
                 p.grad.data.zero_()
         #agent.policy_net.zero_grad() # set gradient to zero 
         #agent.value_net.zero_grad() # set gradient to zero for value network
         
 
         loss = -  (log_probs * A_k).mean() # computes the loss
         loss.backward(retain_graph=True) # compute gradient
         
         loss_V = nn.MSELoss()(V, G) # compute the loss for the value network
         loss_V.backward(retain_graph = True) # compute the gradient for the value network
         
         # save updated parameters for the policy network
         for name, p in param_pol.items():
             new_val_pol = update_function(p, p.grad, loss, learning_rate = 0.003)
             new_val_pol.retain_grad()
             param_pol[name] = new_val_pol
             
         # save updated parameters for the value network
         
         for name, p in param_val.items():
             new_val_val = update_function(p, p.grad, loss_V, learning_rate = 0.003)
             new_val_val.retain_grad()
             param_val[name] = new_val_val           
         
     # Run adapted policy
     A_k, av_reward, log_probs, T_batch_states, T_batch_actions, V, G, av_lengths_traj =  agent.sample_traj(env, hyper_param.K, hyper_param.H, param_pol, param_val)
     env.close()
     
     
     log_probs_old = log_probs.detach().clone()
     
     # compute metaloss (ppo) for this iteration
     log_probs, entropy = agent.prob(T_batch_states, T_batch_actions, param_pol) 
     ratios = torch.exp(log_probs - log_probs_old)
     surr1 = ratios * A_k
     surr2 = torch.clamp(ratios, 1 - hyper_param.clip, 1 + hyper_param.clip) * A_k
     adapted_loss = (- torch.min(surr1, surr2)).mean() - 0.005*entropy.mean()            
     
    
     # computed mataloss for value network
     adapted_loss_V = nn.MSELoss()(V, G)
     
     return(adapted_loss, adapted_loss_V, av_reward, av_lengths_traj)
 
    
def hyperparameters(algo = "pg"):
    learning_rate = 0.003
    learning_rate_value_net = 0.003 # learning rate for the value network
    gamma = 1
    TARGET_UPDATE = 1 # number of epochs between two print statements
    UPDATE_V = 1 # number of epochs between two value function updates
    nb_updates_per_epoch = 1
    clip = 0.2
    if algo == "pg":
        pass
    elif algo == "pgb":
        learning_rate =   0.003
        learning_rate_value_net =  0.003 # learning rate for the value network
        TARGET_UPDATE = 1 # number of epochs between two print statements
        UPDATE_V = 1 # number of epochs between two value function updates
    elif algo == "ppo":
        learning_rate =   0.003
        TARGET_UPDATE = 10 # number of epochs between two print statements
        UPDATE_V = 1 # number of epochs between two value function updates
        nb_updates_per_epoch = 10
    
    return(learning_rate, learning_rate_value_net, gamma, TARGET_UPDATE, UPDATE_V, nb_updates_per_epoch, clip )


def scaling(tensor): #this function scales the rewards by substracting the mean and diving by the standard deviation
        return((tensor - tensor.mean()) /(tensor.std()+ 1e-10))


            
            
def plot_learning(buffer_ls, nb_steps = 500000): # takes a list of OnPolicyBuffer after learning and plot the results

    buffer = buffer_ls[0]
    fig, ax = plt.subplots(2, sharex= True)
    fig.suptitle("Policy-based learning in " + buffer.env )
    ax[0].set_title("Average rewards per epoch")
    ax[1].set_xlabel("Number of steps")
    ax[1].set_title("Moving Average of rewards per epoch")
    
    for buffer in buffer_ls:
        steps = list( buffer.cum_steps)[:nb_steps]
        av_rewards = list(buffer.av_rewards)[:nb_steps]
        mov_av_rewards = list(buffer.mov_av_rewards)[:nb_steps]
        algo = buffer.algo
        ax[0].plot(steps, av_rewards, label = algo)
        ax[1].plot(steps, mov_av_rewards, label = algo)
    
    ax[0].legend()
    ax[1].legend()
    #plt.show()
    plt.savefig("myplot.png")
    
def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    
def update_function(param, grad, loss, learning_rate =0.1):
    return(param - learning_rate * grad)