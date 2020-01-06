import numpy as np
from torch import nn
import torch
from torch.optim import Adam
from models import charRNN
from datareader import load_dataset, get_vocab_size
from constants import *
from ops import prep_batch

# load data
training_data = load_dataset(DATASET_NAME)
data_size = training_data.shape[0]

def get_batch(batch_size):
	rand_ind = np.random.randint(data_size,size=batch_size)
	batch = training_data[rand_ind]
	
	return prep_batch(batch)

# get model ready
model = charRNN(get_vocab_size())

# Training loop
for ITER in range(ITERATIONS):


