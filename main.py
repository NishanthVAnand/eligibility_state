import gym
import numpy as np
import matplotlib.pyplot as plt
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("/Volumes/MyStuff/new_env_gym/")
import pomdp_grid_world

from select_action import getAction
from getQ import getQValue

parser = argparse.ArgumentParser(description='eligibility trace on states')
parser.add_argument('--env', default="pomdpGridWorld-v0", help="Environment")
parser.add_argument('--eps', type=float, default=0.9, help="Exploration rate at the beginning")
parser.add_argument('--n_epi', type=int, default=500, help="Number of episodes in an experiment")
parser.add_argument('--lr_rate', type=float, default=0.0005, help="learning rate to update Q-values")
parser.add_argument('--trace', type=float, default=0.9, help="trace parameter (lambda)")
parser.add_argument('--gamma', type=float, default=0.9, help="discount parameter")
parser.add_argument('--n_exp', type=int, default=50, help="number of independent experiments")
parser.add_argument('--FA_type', default="OneLinearNet", help="Type of Function Approximator")

args = parser.parse_args()

env = gym.make(args.env)

# choosing the type of function approximator
if args.FA_type == "OneLinearNet":
    from FA import OneLinearNet
    net = OneLinearNet(len(env.observation_space.sample()), env.action_space.n)

findMode = lambda x: "explore" if np.random.uniform(0,1) < x else "exploit"

def experiment(env, gamma, trace, eps, n_epi, lr_rate):
    """
    Performs one independent experiment.
    Arguments:
    1. env - gym environment
    2. gamma - discount parameter
    3. trace - eligibilty trace parameter
    4. eps - exploration rate
    5. n_epi - number of episodes in one experiment
    6.lr_rate - learning rate

    Return:
    A list of rewards received in each episode/trial
    """
    all_rewards = []

    for epi in range(n_epi): # Number of trials
        curr_obs = env.reset()
        curr_state = np.array(curr_obs) # defining state from observation
        done = False
        curr_action = getAction(curr_state, findMode(eps), env, net)
        epi_rew = 0
        while not done:
            next_obs, reward, done, info = env.step(curr_action)
            curr_value = getQValue(curr_state, net)[curr_action]
            net.zero_grad()
            grad = curr_value.backward()
            
            # Calculating TD error
            if done:
                td_error = (reward - curr_value)
            else:
                next_state = trace * curr_state + np.array(next_obs) # State: defining eligibilty trace state
                next_action = getAction(next_state, findMode(eps), env, net)
                next_value = getQValue(next_state, net)[next_action]
                td_error = reward + (gamma*next_value) - curr_value
            
            # Updating the network parameters
            with torch.no_grad():
                    for param in net.parameters():
                        param.data += lr_rate * td_error * param.grad
            
            curr_state = next_state
            curr_action = next_action

            epi_rew += reward
        
        # decaying exploration rate every episode
        if eps >= 0.05:
            eps -= 0.005
        
        all_rewards.append(epi_rew)

if __name__ == "__main__":
    exp_rew = []
    for exp in range(args.n_exp):
        exp_rew.append(experiment(env, args.gamma, args.trace, args.eps, args.n_epi, args.lr_rate))