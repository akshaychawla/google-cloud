#!/bin/bash

echo "Installing PIP + upgrading"
sudo apt-get install python-pip
sudo apt-get install python-imaging
sudo apt-get install htop
pip install --upgrade pip

echo "Installing Python libs"
pip install --user keras tqdm opencv-python h5py

echo "Setting keras and theano settings"
mkdir ~/.keras
cp .theanorc ~/
cp keras.json ~/.keras/

echo "Installing 7z tool"
sudo apt-get install p7zip-full p7zip

echo "Downloading train.7z + test.7z"
wget --load-cookies cookies.txt -c https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/download/train.7z
wget --load-cookies cookies.txt -c https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/download/test.7z

echo "Unzipping train.7z + test.7z"
7z x ./train.7z 
7z x ./test.7z

echo "removing CUDA and train.7z"
rm -rf ./cuda* 
rm -rf ./train.7z

echo "Downloading VGG16 pre-trained - THEANO VERSION PLS NOTE"
wget -c https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5

echo "Installing cudnn"
wget --load-cookies cookies.txt -c https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v5.1/prod_20161129/8.0/cudnn-8.0-linux-x64-v5.1-tgz

echo "Downloading RESNET 50 THEANO version weights - ImageNet"
wget -c https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5