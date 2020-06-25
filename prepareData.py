# MATHEW LEWIS, JUNE 2020

import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(len(x_train),28*28)
x_test = x_test.reshape(len(x_test),28*28)
y_train,y_test = y_train,y_test


np.save('data/x_test.npy',x_test)
np.save('data/y_test.npy',y_test)
np.save('data/x_train.npy',x_train)
np.save('data/y_train.npy',y_train)
