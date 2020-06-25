# MATHEW LEWIS, JUNE 2020

import numpy as np
from tensorflow.keras.models import model_from_json
import tensorflow as tf
tf.keras.backend.set_floatx('float64')


def save(model,fileName):
	open(fileName+".json","w").write(model.to_json())
	model.save_weights(fileName+".h5")

def compAndTrain(model,x_train,y_train,epochs):
	# compile the model
	model.compile(optimizer='adam',
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=['accuracy'])
	# train the model
	model.fit(x_train, y_train, epochs=epochs)

def load(fileName):
	# load json and create model
	model = model_from_json(open(fileName+'.json','r').read())
	# load weights into new model
	model.load_weights(fileName+'.h5')
	# compile the model
	model.compile(loss='binary_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
	return model

def run(model,input):
	runable_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
	L = [list(l) for l in np.array(runable_model(input))]
	return [x.index(max(x)) for x in L]





