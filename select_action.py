import numpy as np
import torch

from getQ import getQValue

def getAction(state, mode, env, net):
    """
    This function selects the action according to epsilon-greedy policy
    Takes two arguments:
    1. state - a numpy vector for the state representation
    2. mode - to "explore" or to "exploit"

    returns the action to be picked
    """
    q_values = getQValue(state, net) 

    if mode == "explore":
        return env.action_space.sample()

    elif mode =="exploit" and (torch.max(q_values) == q_values).sum() > 1:
        q_values_np = q_values.numpy()
        idx = np.argwhere(q_values_np == np.max(q_values_np))
        return idx[np.random.randint(0,idx.shape[0])][0]

    else:
        return torch.argmax(q_values).item()