import numpy as np
from datagen import * 
from modelBuilder import *

def TEST_datagen():

	for mytup in generator(debug=False):
		assert mytup is not None

def TEST_LeNet():

	model = LeNet()
	print model.summary()



if __name__=="__main__":
	# TEST_datagen()
	TEST_LeNet()