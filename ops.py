from constants import *
import numpy as np
import torch
from torch import nn
from torch.nn.functional import F

def prep_batch(batch):
	# Batch comes in as numpy array
	batch = torch.from_numpy(batch).float()
	if USE_CUDA: batch = batch.cuda()
	return batch

# Convert model output to one hot vector
def one_hot_from_output(T):
	new_T = torch.zeros_like(T)

	# If batch
	if len(list(T.shape)) == 2:
		for i,t in enumerate(T):
			new_T[i][int(torch.argmax(t))] = 1

	# Single tensor
	elif len(list(T.shape)) == 1:
		new_T[int(torch.argmax(T))] = 1

	return new_T

