import numpy as np
from torch import nn
import torch
from torch.optim import Adam
from models import charRNN
from datareader import load_dataset, get_vocab_size
import ops
from constants import *
from ops import prep_batch

# load data
training_data = load_dataset(DATASET_NAME)
data_size = training_data.shape[0]

# Returns BATCH_SIZE sequences of SEQ_LENGTH
def get_batch():
	rand_ind = np.random.randint(data_size-SEQ_LENGTH,size=BATCH_SIZE)
	batch = np.asarray([training_data[rand_ind[i]:rand_ind[i]+SEQ_LENGTH] for i in range(BATCH_SIZE)])
	
	return prep_batch(batch)

# get model ready
model = charRNN(get_vocab_size())
opt = Adam(model.parameters(),lr=LEARNING_RATE)
LF = nn.MSELoss()

# Training loop
for ITER in range(ITERATIONS):
	opt.zero_grad()

	batch = get_batch().squeeze().unsqueeze(1)
	Y = []
	
	h = model.initH()
	for i in range(SEQ_LENGTH-1):
		y,h = model(batch[i],h)
		Y.append(y)
	Y = torch.cat(Y)
	loss = LF(Y.unsqueeze(1),batch[1:])
	loss.backward()
	opt.step()

	print(ops.one_hot_to_string(ops.one_hot_from_output(Y)))
	

