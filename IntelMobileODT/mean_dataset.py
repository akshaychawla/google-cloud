''' this script calculated the mean of the dataset incrementaly '''

import ConfigParser
import cv2
import numpy as np 
import os, sys
from datagen import load_filenames_labels

config = ConfigParser.RawConfigParser()
config.read("./defaults.cfg")

TRAIN_DATA_LOCATION 	= config.get("data_folder", "TRAIN_DATA_LOCATION") 

fnames_labels = load_filenames_labels()

running_mean  = 0.0
current_index = 0
for fname,_ in enumerate(fnames_labels):

