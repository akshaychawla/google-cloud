''' This script splits the training data into train and validation splits '''

import shutil
import os, sys 
from datagen import load_filenames_labels, shuffle_fnames_labels
from tqdm import *
import ConfigParser

config = ConfigParser.RawConfigParser()
config.read("./defaults.cfg")

# globals 
TRAIN_DATA_LOCATION 	= config.get("data_folder", "TRAIN_DATA_LOCATION")
NUM_VALID 				= config.getint("training", "NUM_VALID")
VALID_DATA_LOCATION   	= config.get("data_folder", "VALID_DATA_LOCATION")
REPLACEMENT 			= config.getint("training", "REPLACEMENT")

print TRAIN_DATA_LOCATION, NUM_VALID, VALID_DATA_LOCATION, REPLACEMENT

def main():

	# Get traning file names
	fnames_labels 		= shuffle_fnames_labels(load_filenames_labels())
	fnames_labels_valid = fnames_labels[0:NUM_VALID]

	# create validation folder 
	os.mkdir(VALID_DATA_LOCATION)
	for dat_type in ("Type_1", "Type_2", "Type_3"): os.mkdir(VALID_DATA_LOCATION+dat_type)

	# copy files to validation folder
	for fname_t,_ in tqdm(fnames_labels_valid):
		fname_v = fname_t.replace('train','valid')
		shutil.copy2(fname_t, fname_v)

		# if replacement is False , remove fname_t image (as it has been copied to the validation folder) 
		if REPLACEMENT == 0:
			os.remove(fname_t)



if __name__ == '__main__':
 	main() 