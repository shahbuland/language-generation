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
	seqs = [[training_data[i+j] for i in rand_ind] for j in range(SEQ_LENGTH)]
	return prep_batch(np.asarray(seqs))

# get model ready
model = charRNN(get_vocab_size())
if USE_CUDA: model.cuda()
opt = Adam(model.parameters(),lr=LEARNING_RATE)
LF = nn.MSELoss()

# Training loop
for ITER in range(ITERATIONS):
	opt.zero_grad()

	batch = get_batch()
	y = model(batch[:-1])	
	loss = LF(y,batch[1:])
	loss.backward()
	nn.utils.clip_grad_norm_(model.parameters(), 5)
	opt.step()

	# randomly sample one sequence
	sample = y[:,np.random.randint(BATCH_SIZE),:]
	print(ops.one_hot_to_string(ops.one_hot_from_output(sample)))

