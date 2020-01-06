# Training
SEQ_LENGTH = 50
ITERATIONS = 2000
USE_CUDA = False
# Model
LEARNING_RATE = 2e-3
LR_DECAY = 0.97
LR_DECAY_STEP = 10 # Delay after this many iters
DECAY_RATE = 0.95
GRAD_CLIP = 5
HIDDEN_SIZE = 128
# Data
DATASET_NAME = "nosleep.txt"
