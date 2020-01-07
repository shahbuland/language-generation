# This file is meant to convert text files into readable data

import numpy as np

# Returns string of entire file
def read_file(dataset_name):
	f = open("./raw_datasets/"+dataset_name,"r")
	return f.read()

def get_vocab():
	f = open("vocab.txt","r")
	vocab = f.read()
	vocab = vocab.split(" ")
	vocab[-1] = vocab[-1][0] # remove \n from last char
	vocab.append(" ") # add space
	return vocab

# Converts string to one hot array
# Only basic punctuation is preserved,
# all else is converted to a space
# capitalization is ignored
# uses vocab
def string_to_1_hot_array(s):
	vocab = get_vocab()
	vocab_size = len(vocab)
	dataset = [] # Datast intially stored as list
	# Now we use vocab to convert s into a sequence of vectors
	# Iterate through characters
	for c in s:
		v = np.zeros((vocab_size))
		try:
			ind = vocab.index(c)
			v[ind] = 1
		except:
			continue
		dataset.append(v)

	# Finally dataset is converted to array
	return np.asarray(dataset)

# Takes name of dataset, finds in raw_datasets folder, converts to model ready dataset, puts in datasets folder
def txt_to_npy(dataset_name):
	s = read_file(dataset_name)
	dataset = string_to_1_hot_array(s)
	np.save("./datasets/"+dataset_name, dataset)

# Loads and returns dataset of certain name
def load_dataset(dataset_name):
	return np.load("./datasets/"+dataset_name+".npy")

# Simply gets rise of vocab
def get_vocab_size():
	return len(get_vocab())
