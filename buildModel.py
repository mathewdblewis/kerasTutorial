# MATHEW LEWIS, JUNE 2020

import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from utilities import save,compAndTrain

# get the data
x_train = np.load('data/x_train.npy')
y_train = np.load('data/y_train.npy')



# build the model

# this initializes the model
model = tf.keras.models.Sequential()
# this adds a layer with 128 nodes, dense means they are all connected
model.add(tf.keras.layers.Dense(128, activation='relu'))
# dropout means when training we drop out some of the edges in the graph
# this makes the model more robust
model.add(tf.keras.layers.Dropout(0.2))
# another 10 nodes (10 because the demo classifies then numbers 1 through 10)
model.add(tf.keras.layers.Dense(10))

# compile and train the model
compAndTrain(model,x_train,y_train,4)
# save the model
save(model,'models/model')










