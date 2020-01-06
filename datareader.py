# This file is meant to convert text files into readable data

import numpy as np

# Returns string of entire file
def read_file(dataset_name):
	f = open("./raw_datasets/"+dataset_name,"r")
	return f.read()

# Converts string to one hot array
# Only basic punctuation is preserved,
# all else is converted to a space
# capitalization is ignored
# uses vocab
def string_to_1_hot_array(s):
	f = open("vocab.txt","r")
	vocab = f.read()
	vocab = vocab.split(" ") # array of characters
	vocab[-1] = vocab[-1][0] # last char will have \n, so we remove it
	vocab.append(" ") # Add space key
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
	return np.load("./datasets/"+dataset_name)
