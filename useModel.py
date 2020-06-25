# MATHEW LEWIS, JUNE 2020

import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from utilities import run,load

# load the test data
x_test = np.load('data/x_test.npy')
y_test = np.load('data/y_test.npy')

# run the model on some test data
result = run(load('models/model'),x_test[:10])

# compare the model's output with the correct answer
print('correct answer:\t\t',list(y_test[:10]))
print('models output:\t\t',result)
