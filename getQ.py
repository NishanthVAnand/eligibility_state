import torch

def getQValue(state, net):
	"""
	Takes two inputs
	1. net object - function approximation network
	2. state - a numpy vector to represent the state
	Return
	The q-values corresponding to all the actions in that state
	"""
	return net.forward(torch.from_numpy(state).type(torch.FloatTensor))