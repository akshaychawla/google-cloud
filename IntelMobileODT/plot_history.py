''' This script displays the Training acc, loss and validation acc, loss accumulated during train time ''' 

import cPickle as pickle 
import matplotlib.pyplot as plt 
import numpy as np


history = None 

with open("history.pkl","rb") as f:
	history = pickle.load(f)
	f.close() 


train_acc  = history["acc"]
train_loss = history["loss"]
val_acc    = history["val_acc"]
val_loss   = history["val_loss"]

plt.subplot(121)
plt.plot(np.arange(len(train_acc)), train_acc, "r", label="training accuracy")
plt.plot(np.arange(len(val_acc)), val_acc, "g", label="validation accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.subplot(122)
plt.plot(np.arange(len(train_loss)), train_loss, "r", label="training loss")
plt.plot(np.arange(len(val_loss)), val_loss, "g", label="validation loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()