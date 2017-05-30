import cv2 
import numpy as np
import os, sys
from keras.utils import to_categorical 
import ConfigParser

config = ConfigParser.RawConfigParser()
config.read("./defaults.cfg")

TRAIN_DATA_LOCATION 	= config.get("data_folder", "TRAIN_DATA_LOCATION")

def load_filenames_labels():

	type_1_fnames = list(map(lambda x: TRAIN_DATA_LOCATION+"Type_1/"+x, os.listdir(TRAIN_DATA_LOCATION+"Type_1/")))
	type_2_fnames = list(map(lambda x: TRAIN_DATA_LOCATION+"Type_2/"+x, os.listdir(TRAIN_DATA_LOCATION+"Type_2/")))
	type_3_fnames = list(map(lambda x: TRAIN_DATA_LOCATION+"Type_3/"+x, os.listdir(TRAIN_DATA_LOCATION+"Type_3/")))

	fnames_labels = []

	for fname in type_1_fnames: fnames_labels.append([fname, 0])
	for fname in type_2_fnames: fnames_labels.append([fname, 1])
	for fname in type_3_fnames: fnames_labels.append([fname, 2])

	return fnames_labels

def shuffle_fnames_labels(fnames_labels):

	# shuffle indices
	indices  = np.arange(len(fnames_labels))
	np.random.shuffle(indices)

	# shuffle filenames
	fnames_labels_shuffled = []
	for idx in indices:
		fnames_labels_shuffled.append(fnames_labels[idx])

	del fnames_labels
	
	return fnames_labels_shuffled

# def generator(mini_batch = 3, debug = False):

# 	fnames_labels = load_filenames_labels()
# 	fnames_labels = shuffle_fnames_labels(fnames_labels)
	
# 	if debug:
# 		print "...fnames and labels loaded"

# 	while 1:
# 		for i in xrange(0,len(fnames_labels), mini_batch):

# 			j = min(len(fnames_labels), i+mini_batch)

# 			# minibatch labels and filenames
# 			fnames_labels_mbatch = fnames_labels[i:j]
			
# 			if debug:
# 				print i,j
# 				print fnames_labels_mbatch

# 			# load images in fnames_mbatch
# 			x,y = [],[]
# 			for fname,label in fnames_labels_mbatch:
# 				img = cv2.imread(fname)
# 				if RESIZE:
# 					img = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT))
# 				img = np.rollaxis(img, 2)
# 				x.append(img)
# 				y.append(label)

# 			x = np.array(x)
# 			y = to_categorical(np.array(y), num_classes=3)
			
# 			if debug:
# 				print x.shape, y.shape

# 			yield tuple([x,y])




	


