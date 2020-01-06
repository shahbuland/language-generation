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
