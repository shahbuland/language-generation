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

# Returns BATCH_SIZE sequences of SEQ_LENGTH
def get_batch():
	rand_ind = np.random.randint(data_size-SEQ_LENGTH,size=BATCH_SIZE)
	batch = np.asarray([training_data[rand_ind[i]:rand_ind[i]+SEQ_LENGTH] for i in range(BATCH_SIZE))
	
	return prep_batch(batch)

# get model ready
model = charRNN(get_vocab_size())

# Training loop
for ITER in range(ITERATIONS):


