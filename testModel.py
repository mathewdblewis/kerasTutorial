import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from utilities import run,load

# load the test data
x_test = np.load('data/x_test.npy')
y_test = np.load('data/y_test.npy')

# test the model
model = load('models/model')
score = model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))