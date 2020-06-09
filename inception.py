from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import models
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, Activation, MaxPooling2D, BatchNormalization
from keras import optimizers, regularizers
from keras.optimizers import SGD

class Inception(object):
	def __init__(self, model_path='models/inception_weights_best.h5'):
		self.reconstructed_model = models.load_model(model_path)

	def read_image(self, path):
	    image = cv2.imread(path, cv2.IMREAD_COLOR)
	    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	    return image

	def __get_labels(self):
		labels = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
		col = {'c0': 'safe driving',
		'c1': 'texting - right',
		'c2': 'talking on the phone - right',
		'c3': 'texting - left',
		'c4': 'talking on the phone - left',
		'c5':'operating the radio',
		'c6': 'drinking',
		'c7': 'reaching behind',
		'c8': 'hair and makeup',
		'c9': 'talking to passenger'}
		return labels, col

	def prediction(self, image_path):
		labels, col = self.__get_labels()
		image = load_img(path=image_path, color_mode="grayscale",
	                                              target_size=(160, 120))
		input_arr = img_to_array(image)
		input_arr = np.array([input_arr])  # Convert single image to a batch.
		predictions = self.reconstructed_model.predict(input_arr)
		print(predictions)
		
		predict = col[labels[np.argmax(predictions[0])]]
		score = 0
		return predict, score	