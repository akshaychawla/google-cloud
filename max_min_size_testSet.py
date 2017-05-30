''' This script finds the maximum and minimum sized images in the TESTSET '''

import numpy as np
import cv2 
import os
from tqdm import *

TESTSET_location = "./test/"

def main():

	filenames = os.listdir(TESTSET_location)

	min_height, min_width = 9000,9000
	max_height, max_width = 0,0

	for fname in tqdm(filenames):

		img = cv2.imread(TESTSET_location+fname)
		h,w = img.shape[0:2]

		if h<min_height:
			min_height = h 
		if h>max_height:
			max_height = h 

		if w<min_width:
			min_width = w
		if w>max_width:
			max_width = w

	print "Min Height: %d  -- Min Width: %d -- Max Height: %d -- Max Width: %d" % (min_height,min_width,max_height,max_width)


if __name__ == '__main__':
	main()