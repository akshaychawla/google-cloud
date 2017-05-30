from modelBuilder import LeNet, tinyVGG, VGG16_network, VGG16_1
import numpy as np
import cv2
# import matplotlib.pyplot as plt 
from tqdm import *
from collections import Counter
import argparse
from keras.preprocessing.image import ImageDataGenerator 
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import SGD
from math import floor, ceil
from datagen import load_filenames_labels
from keras.callbacks import ModelCheckpoint
import ConfigParser

config = ConfigParser.RawConfigParser()
config.read("./defaults.cfg")

TRAIN_DATA_LOCATION 	= config.get("data_folder", "TRAIN_DATA_LOCATION")
BATCH_SIZE 				= config.getint("training", "BATCH_SIZE")
VALID_DATA_LOCATION   	= config.get("data_folder", "VALID_DATA_LOCATION")
IMAGE_SIZE				= ( config.getint("data_process","IMAGE_HEIGHT"), config.getint("data_process","IMAGE_WIDTH") )
NUM_EPOCHS				= config.getint("training", "NUM_EPOCHS")

print TRAIN_DATA_LOCATION, BATCH_SIZE, VALID_DATA_LOCATION, IMAGE_SIZE

def preprocess_fxn(img):
	''' This fxn takes a 3-dim numpy tensor img (RGB) and performs ops: 
		1. center crop 
		2. resize to IMAGE_SIZE
		2. [?] imagenet means subtraction 
		return 3-dim numpy tensor
	'''
	# convert from (3,h,w) to (h,w,3)
	img = np.rollaxis(img, axis=2)
	img = np.rollaxis(img, axis=2)

	# Centre - crop if height>width
	height, width 	= img.shape[0:2]
	center_crop     = img
	if height > width:
		center_height 	= int(height / 2)
		crop_lowerlimit = int(center_height - ceil(width/2))
		crop_upperlimit = int(center_height + floor(width/2))
		center_crop     = img[crop_lowerlimit:crop_upperlimit, :, :]

	# resize to IMAGE_SIZE
	# img_resize 		= cv2.resize(center_crop, (IMAGE_SIZE[1], IMAGE_SIZE[0])).astype("float32") 

	# imageNet means subtraction
	# img_resize[:,:,0] -= 123.680 # R
	# img_resize[:,:,1] -= 116.779 # G
	# img_resize[:,:,2] -= 103.939 # B

	# convert from (h,w,3) to (3,h,w)
	center_crop = np.rollaxis(center_crop, axis=2)

	return center_crop



# TO PREVENT ERRORS IN PIL 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from datagen import load_filenames_labels


def main():

	# get model 
	# model = tinyVGG()
	model = VGG16_1()
	print model.summary()

	# TRAIN DATA generator
	_train_datagen 	= ImageDataGenerator(
						preprocessing_function=preprocess_fxn,
						rescale=1./255,
						horizontal_flip=True,
						vertical_flip=True,
						rotation_range=20
					)

	train_datagen 	= _train_datagen.flow_from_directory(
						TRAIN_DATA_LOCATION,
						target_size=IMAGE_SIZE,
						batch_size=int(BATCH_SIZE)
						)

	# VALID DATA generator
	_valid_datagen 	= ImageDataGenerator(
							rescale = 1./255,
							preprocessing_function=preprocess_fxn
						)
	valid_datagen 	= _valid_datagen.flow_from_directory(
						VALID_DATA_LOCATION,
						target_size=IMAGE_SIZE,
						batch_size=int(BATCH_SIZE)
						)

	# get total number of training images
	_num_train = train_datagen.samples
	_num_valid = valid_datagen.samples
	print "Valid samples calculated: ",valid_datagen.samples, "Train samples calc:", train_datagen.samples
	
	# model checkpointing 
	val_acc_filepath="./snapshots/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
	val_loss_filepath="./snapshots/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
	val_acc_checkpoint = ModelCheckpoint(val_acc_filepath, monitor="val_acc", 
								 verbose=1, save_best_only=True, mode="max")
	val_loss_checkpoint = ModelCheckpoint(val_loss_filepath, monitor="val_loss", 
								 verbose=1, save_best_only=True, mode="min")
	

	filename_labels = load_filenames_labels()
	labels 			= [val[1] for val in filename_labels]
	

	# fit model to data
	model.fit_generator(train_datagen, 
						steps_per_epoch=ceil(float(_num_train)/BATCH_SIZE), 
						epochs=NUM_EPOCHS,
						validation_data=valid_datagen,
						validation_steps=ceil(float(_num_valid)/BATCH_SIZE),
						callbacks=[val_acc_checkpoint, val_loss_checkpoint]
						)

	# save the last one 
	model.save('./snapshots/model.final.h5')


if __name__ == "__main__":
	main()


