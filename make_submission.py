''' This script generates a submission csv file that can be submitted '''

import numpy as np 
import cv2 
import csv 
import os, sys 
from tqdm import *
import ConfigParser
from keras.models import load_model
from math import ceil, floor


def preprocess_fxn(img):
	''' This fxn takes a 3-dim numpy tensor img (RGB) and performs ops: 
		1. center crop 
		2. resize to IMAGE_SIZE
		2. [?] imagenet means subtraction 
		return 3-dim numpy tensor
	'''

	# Centre - crop if height>width
	height, width 	= img.shape[0:2]
	center_crop     = img
	if height > width:
		center_height 	= int(height / 2)
		crop_lowerlimit = int(center_height - ceil(width/2))
		crop_upperlimit = int(center_height + floor(width/2))
		center_crop     = img[crop_lowerlimit:crop_upperlimit, :, :]

	return center_crop

IMAGE_SIZE = (299, 299)
TEST_DATA_LOCATION = "../test/"

# load filenames
filenames = os.listdir(TEST_DATA_LOCATION)
print "Found : {} test images".format(len(filenames))

# load model 
model = load_model("weights-improvement-99-Accu-97.71.hdf5")
print model.summary()

# make predictions
predictions = []
for fname in tqdm(filenames):
	
	# load and process img
	img = cv2.imread(TEST_DATA_LOCATION+fname)
	img = preprocess_fxn(img)
	img = cv2.resize(img, dsize=(IMAGE_SIZE[1], IMAGE_SIZE[0]))
	img = np.rollaxis(img, axis=2)
	img = np.expand_dims(img, axis=0)
	img = img[:, ::-1, :, :] # NOTE CONVERT BGR img (as loaded by cv2) to RGB format (as loaded by ImageDataGen in training)
	img = img/255.0 # SCALE
	# predict 
	op = model.predict(img)[0]
	# print op
	predictions.append([fname,op])


# sort predictions based on image name 
predictions = sorted(predictions, key=lambda x: int(x[0].strip(".jpg")))
print list(map(lambda x: x[0], predictions))

# dump to disk 
with open("subm_epoch_99_aloss_9771","wb") as f:
	writer = csv.writer(f)
	writer.writerow(["image_name","Type_1","Type_2","Type_3"])
	for fname,pred in predictions:
		writer.writerow([fname, pred[0], pred[1], pred[2]])
	f.close()



