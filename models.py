from torch import nn
import torch
from torch.nn import functional as F
from constants import *

class charRNN(nn.Module):
	def __init__(self, vocab_size):
		super(charRNN, self).__init__()
		
		# Input is [vocab_size] vector
		self.fcx = nn.Linear(vocab_size+HIDDEN_SIZE,vocab_size)
		self.fch = nn.Linear(vocab_size+HIDDEN_SIZE,HIDDEN_SIZE)

		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self,x,h):
		comb = torch.cat((x,h),1)
		h_next = self.fch(comb)
		out = self.fcx(comb)
		return out,h_next

	def initH(self):
		return torch.zeros(1,HIDDEN_SIZE)
			
