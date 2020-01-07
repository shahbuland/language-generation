from torch import nn
import torch
from torch.nn import functional as F
from constants import *

class charRNN(nn.Module):
	def __init__(self, vocab_size):
		super(charRNN, self).__init__()
		
		# Input is [vocab_size] vector
		self.layers = nn.RNN(vocab_size,HIDDEN_SIZE,LAYERS,nonlinearity="tanh")
		self.output = nn.Linear(HIDDEN_SIZE,vocab_size)
		self.softmax = nn.LogSoftmax()
	def forward(self,x):
		
		h = torch.zeros(LAYERS,BATCH_SIZE,HIDDEN_SIZE)
		if USE_CUDA: h = h.cuda()
		y,_ = self.layers(x,h)
		y = self.softmax(self.output(y))
		return y
