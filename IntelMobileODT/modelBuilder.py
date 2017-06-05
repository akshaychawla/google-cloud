import numpy as np 
# import matplotlib.pyplot as plt 
import cv2 
from keras.models import Sequential,Model
from keras.layers import Conv2D, MaxPool2D, Input, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Dense, Dropout, Activation
from keras.utils import layer_utils
from keras.optimizers import SGD
import keras.backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
import ConfigParser


config = ConfigParser.RawConfigParser()
config.read("./defaults.cfg")

IMAGE_SIZE 		= (3,  config.getint("data_process","IMAGE_HEIGHT"), config.getint("data_process","IMAGE_WIDTH") )
VGG16_WTS  		= config.get("training", "VGG16_WTS")
PRETUNED_WTS 	= config.get("training", "PRETUNED_WTS")
RESNET_WTS   	= config.get("training", "RESNET_WTS") 

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


def VGG16_network_BN_FULLTUNE():

	# image input 
	img_input = Input(shape=IMAGE_SIZE)	

	# Block 1
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
	x = BatchNormalization()(x)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
	x = BatchNormalization()(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool')(x)

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
	x = BatchNormalization()(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
	x = BatchNormalization()(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block2_pool')(x)

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
	x = BatchNormalization()(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
	x = BatchNormalization()(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
	x = BatchNormalization()(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block3_pool')(x)

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
	x = BatchNormalization()(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
	x = BatchNormalization()(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
	x = BatchNormalization()(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block4_pool')(x)

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
	x = BatchNormalization()(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
	x = BatchNormalization()(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
	x = BatchNormalization()(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block5_pool')(x)

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

	# compile and solve using sgd	
	sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

	return model

	return model

def VGG16_network_FULLTUNE():

	# image input 
	img_input = Input(shape=IMAGE_SIZE)	

	# Block 1
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool', trainable=False)(x)

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=True)(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=True)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block2_pool', trainable=True)(x)

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=True)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=True)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=True)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block3_pool', trainable=True)(x)

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=True)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=True)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=True)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block4_pool', trainable=True)(x)

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=True)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=True)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=True)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block5_pool', trainable=True)(x)

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

	# compile and solve using sgd 
	sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

	return model

def VGG16_network_FULLTUNE():

	# image input 
	img_input = Input(shape=IMAGE_SIZE)	

	# Block 1
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=True)(img_input)
	x = Batch
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=True)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool', trainable=True)(x)

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=True)(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=True)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block2_pool', trainable=True)(x)

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=True)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=True)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=True)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block3_pool', trainable=True)(x)

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=True)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=True)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=True)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block4_pool', trainable=True)(x)

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=True)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=True)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=True)(x)
	x = MaxPool2D((2, 2), strides=(2, 2), name='block5_pool', trainable=True)(x)

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

	# compile and solve using sgd 
	sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

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

def inception_3_fulltune():

	# create the base pre-trained model
	print "###########GETTING INCEPTION-V3 MODEL##############"
	base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=IMAGE_SIZE)

	print "########### adding classification block ##########"
	# classification block
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dropout(0.3)(x)
	x = Dense(500, activation='relu')(x)
	x = Dropout(0.3)(x)
	predictions = Dense(3, activation='softmax')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	# set learning rate and compile model 
	sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

	return model

def resnet_50_fulltune():

	# base model
	print "#######GETTING RESNET 50 MODEL############"
	base_model = ResNet50(weights=None, include_top=False, input_shape=IMAGE_SIZE)

	# classification block 
	x = base_model.output
	x = Flatten()(x)
	predictions = Dense(3, activation='softmax')(x)

	# this is model we will train 
	model = Model(inputs=base_model.input, outputs=predictions)

	# load weights 
	print "pre-loading resnet50 weights (no top)"
	model.load_weights(RESNET_WTS, by_name=True)

	# set learning rate and compile model 
	sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

	return model

def resnet_50_fulltune_drop():

	# base model
	print "#######GETTING RESNET 50 MODEL############"
	base_model = ResNet50(weights=None, include_top=False, input_shape=IMAGE_SIZE)

	# classification block 
	x = base_model.output
	x = Dropout(0.3)(x)
	x = Flatten()(x)
	x = Dropout(0.3)(x)
	predictions = Dense(3, activation='softmax')(x)

	# this is model we will train 
	model = Model(inputs=base_model.input, outputs=predictions)

	# load weights 
	print "pre-loading resnet50 weights (no top)"
	model.load_weights(RESNET_WTS, by_name=True)

	# set learning rate and compile model 
	sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

	return model


