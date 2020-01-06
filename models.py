from torch import nn
import torch
from torch.nn import functional as F

class charRNN(nn.Module):
	def __init__(self):
		super(charRNN, self).__init__()

		# Input feeds into recurrent layer, which feeds into output
		self.h_layers = nn.RNN(,512,3,
