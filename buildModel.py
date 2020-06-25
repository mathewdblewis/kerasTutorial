# MATHEW LEWIS, JUNE 2020

import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from utilities import save,compAndTrain

# get the data
x_train = np.load('data/x_train.npy')
y_train = np.load('data/y_train.npy')

# build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10))

# compile and train the model
compAndTrain(model,x_train,y_train,4)
# save the model
save(model,'models/model')
