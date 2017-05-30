import numpy as np 
# import matplotlib.pyplot as plt 
import cv2 
from keras.models import Sequential,Model
from keras.layers import Conv2D, MaxPool2D, Input
from keras.layers import Flatten, Dense, Dropout, Activation
from keras.utils import layer_utils
from keras.optimizers import SGD
import keras.backend as K
from keras.applications.resnet50 import ResNet50
import ConfigParser


config = ConfigParser.RawConfigParser()
config.read("./defaults.cfg")

IMAGE_SIZE = (3,  config.getint("data_process","IMAGE_HEIGHT"), config.getint("data_process","IMAGE_WIDTH") )
VGG16_WTS  = config.get("training", "VGG16_WTS")
PRETUNED_WTS = config.get("training", "PRETUNED_WTS")

print IMAGE_SIZE, VGG16_WTS
 

def LeNet():

	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=IMAGE_SIZE, activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))

	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))

	model.add(Flatten()) 
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(3))
	model.add(Activation('softmax'))

	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	return model


def tinyVGG():

	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=IMAGE_SIZE, activation="relu"))
	model.add(Conv2D(32, (3, 3), activation="relu"))
	model.add(MaxPool2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3), activation="relu"))
	model.add(Conv2D(64, (3, 3), activation="relu"))
	model.add(MaxPool2D(pool_size=(2, 2)))

	model.add(Conv2D(128, (3, 3), activation="relu"))
	model.add(Conv2D(128, (3, 3), activation="relu"))
	model.add(MaxPool2D(pool_size=(2, 2)))

	model.add(Conv2D(128, (3, 3), activation="relu"))
	model.add(Conv2D(128, (3, 3), activation="relu"))
	model.add(MaxPool2D(pool_size=(2, 2)))

	model.add(Flatten())	
	model.add(Dense(3))
	model.add(Activation("softmax"))

	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	return model


def VGG16_network():

	# image input 
	img_input = Input(shape=IMAGE_SIZE)	

	# Block 1
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool', trainable=False)(x)

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=False)(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block2_pool', trainable=False)(x)

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=False)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=False)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block3_pool', trainable=False)(x)

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block4_pool', trainable=False)(x)

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block5_pool', trainable=False)(x)

	# classification block
	x = Flatten(name='flatten')(x)
	x = Dense(500, activation='relu', name='fc1_reinit')(x)
	x = Dropout(0.5)(x)
	x = Dense(250, activation='relu', name='fc2_reinit')(x)
	x = Dense(3, activation='softmax', name='predictions_reinit')(x)
	
	model = Model(img_input, x)
	
	# load wts
	print "Pre-loading IMageNet weights", VGG16_WTS
	model.load_weights(VGG16_WTS, by_name=True)

	for layer in model.layers[0:19]:
		print layer.name
		layer.trainable = False

	# set learning rate and compile model 
	# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

	return model

def VGG16_network_FT():

	# image input 
	img_input = Input(shape=IMAGE_SIZE)	

	# Block 1
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool', trainable=False)(x)

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=False)(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block2_pool', trainable=False)(x)

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=False)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=False)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block3_pool', trainable=False)(x)

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block4_pool', trainable=False)(x)

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=True)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=True)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=True)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block5_pool', trainable=True)(x)

	# classification block
	x = Flatten(name='flatten')(x)
	x = Dense(500, activation='relu', name='fc1_reinit')(x)
	x = Dropout(0.5)(x)
	x = Dense(250, activation='relu', name='fc2_reinit')(x)
	x = Dense(3, activation='softmax', name='predictions_reinit')(x)
	
	model = Model(img_input, x)
	
	# load wts
	print "Pre-loading TUNED weights", PRETUNED_WTS
	model.load_weights(PRETUNED_WTS, by_name=True)

	# set learning rate and compile model 
	sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

	return model

def VGG16_0():

	# image input 
	img_input = Input(shape=IMAGE_SIZE)	

	# Block 1
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool', trainable=False)(x)

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=False)(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block2_pool', trainable=False)(x)

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=False)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=False)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block3_pool', trainable=False)(x)

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block4_pool', trainable=False)(x)

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block5_pool', trainable=False)(x)

	# classification block
	x = Flatten(name='flatten')(x)
	x = Dense(500, activation='relu', name='fc1_reinit')(x)
	x = Dense(250, activation='relu', name='fc2_reinit')(x)
	x = Dense(3, activation='softmax', name='predictions_reinit')(x)
	
	model = Model(img_input, x)
	
	# load wts
	print "Pre-loading IMageNet weights", VGG16_WTS
	model.load_weights(VGG16_WTS, by_name=True)

	for layer in model.layers[0:19]:
		print layer.name
		layer.trainable = False

	# set learning rate and compile model 
	# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

	return model

def VGG16_1():

	# image input 
	img_input = Input(shape=IMAGE_SIZE)	

	# Block 1
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool', trainable=False)(x)

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=False)(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block2_pool', trainable=False)(x)

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=False)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=False)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block3_pool', trainable=False)(x)

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block4_pool', trainable=False)(x)

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block5_pool', trainable=False)(x)

	# classification block
	x = Flatten(name='flatten')(x)
	x = Dense(500, activation='relu', name='fc1_reinit')(x)
	x = Dropout(0.1)(x)
	x = Dense(250, activation='relu', name='fc2_reinit')(x)
	x = Dropout(0.1)(x)
	x = Dense(3, activation='softmax', name='predictions_reinit')(x)
	
	model = Model(img_input, x)
	
	# load wts
	print "Pre-loading IMageNet weights", VGG16_WTS
	model.load_weights(VGG16_WTS, by_name=True)

	for layer in model.layers[0:19]:
		print layer.name
		layer.trainable = False

	# set learning rate and compile model 
	# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

	return model

def VGG16_2():

	# image input 
	img_input = Input(shape=IMAGE_SIZE)	

	# Block 1
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool', trainable=False)(x)

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=False)(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block2_pool', trainable=False)(x)

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=False)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=False)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block3_pool', trainable=False)(x)

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block4_pool', trainable=False)(x)

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block5_pool', trainable=False)(x)

	# classification block
	x = Flatten(name='flatten')(x)
	x = Dense(500, activation='relu', name='fc1_reinit')(x)
	x = Dropout(0.2)(x)
	x = Dense(250, activation='relu', name='fc2_reinit')(x)
	x = Dropout(0.2)(x)
	x = Dense(3, activation='softmax', name='predictions_reinit')(x)
	
	model = Model(img_input, x)
	
	# load wts
	print "Pre-loading IMageNet weights", VGG16_WTS
	model.load_weights(VGG16_WTS, by_name=True)

	for layer in model.layers[0:19]:
		print layer.name
		layer.trainable = False

	# set learning rate and compile model 
	# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

	return model

def VGG16_3():

	# image input 
	img_input = Input(shape=IMAGE_SIZE)	

	# Block 1
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool', trainable=False)(x)

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=False)(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block2_pool', trainable=False)(x)

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=False)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=False)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block3_pool', trainable=False)(x)

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block4_pool', trainable=False)(x)

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block5_pool', trainable=False)(x)

	# classification block
	x = Flatten(name='flatten')(x)
	x = Dense(500, activation='relu', name='fc1_reinit')(x)
	x = Dropout(0.3)(x)
	x = Dense(250, activation='relu', name='fc2_reinit')(x)
	x = Dropout(0.3)(x)
	x = Dense(3, activation='softmax', name='predictions_reinit')(x)
	
	model = Model(img_input, x)
	
	# load wts
	print "Pre-loading IMageNet weights", VGG16_WTS
	model.load_weights(VGG16_WTS, by_name=True)

	for layer in model.layers[0:19]:
		print layer.name
		layer.trainable = False

	# set learning rate and compile model 
	# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

	return model

def VGG16_4():

	# image input 
	img_input = Input(shape=IMAGE_SIZE)	

	# Block 1
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool', trainable=False)(x)

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=False)(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block2_pool', trainable=False)(x)

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=False)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=False)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block3_pool', trainable=False)(x)

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block4_pool', trainable=False)(x)

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block5_pool', trainable=False)(x)

	# classification block
	x = Flatten(name='flatten')(x)
	x = Dense(500, activation='relu', name='fc1_reinit')(x)
	x = Dropout(0.4)(x)
	x = Dense(250, activation='relu', name='fc2_reinit')(x)
	x = Dropout(0.4)(x)
	x = Dense(3, activation='softmax', name='predictions_reinit')(x)
	
	model = Model(img_input, x)
	
	# load wts
	print "Pre-loading IMageNet weights", VGG16_WTS
	model.load_weights(VGG16_WTS, by_name=True)

	for layer in model.layers[0:19]:
		print layer.name
		layer.trainable = False

	# set learning rate and compile model 
	# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

	return model

def VGG16_5():

	# image input 
	img_input = Input(shape=IMAGE_SIZE)	

	# Block 1
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool', trainable=False)(x)

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=False)(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block2_pool', trainable=False)(x)

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=False)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=False)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block3_pool', trainable=False)(x)

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block4_pool', trainable=False)(x)

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block5_pool', trainable=False)(x)

	# classification block
	x = Flatten(name='flatten')(x)
	x = Dense(500, activation='relu', name='fc1_reinit')(x)
	x = Dropout(0.5)(x)
	x = Dense(250, activation='relu', name='fc2_reinit')(x)
	x = Dropout(0.5)(x)
	x = Dense(3, activation='softmax', name='predictions_reinit')(x)
	
	model = Model(img_input, x)
	
	# load wts
	print "Pre-loading IMageNet weights", VGG16_WTS
	model.load_weights(VGG16_WTS, by_name=True)

	for layer in model.layers[0:19]:
		print layer.name
		layer.trainable = False

	# set learning rate and compile model 
	# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

	return model