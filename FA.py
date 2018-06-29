import torch
import torch.nn as nn
import torch.nn.functional as F

class OneLinearNet(nn.Module):
	"""
	One Linear Layered Network.
	Takes two arguments:
	1. Input dimension
	2. Output dimension
	"""
	def __init__(self, state_size, action_size):
	    super(OneLinearNet, self).__init__()
	    self.fc = nn.Linear(state_size, action_size, bias=False)

	def forward(self, x):
	    x = self.fc(x)
	    return x