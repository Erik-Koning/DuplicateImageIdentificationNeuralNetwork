# This file contains the tensorflow parts of the code, seperate from the GUI.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import tensorflow as tf
import numpy as np

from model_fn import *

estimator = None

def init_estimator():
	config = tf.contrib.learn.RunConfig(gpu_memory_fraction=0.9)
	
	global estimator

	estimator = tf.contrib.learn.Estimator(
		model_fn = cnn_model_fn,
		model_dir = os.path.join("..", 'data', 'generated_network'),
		config = config
		)

# Combines the two imported images into one numpy.ndarray with 6 "channels"
def combine_images(im1, im2):
	combined = np.array([im1, im2], dtype=np.uint8)
	transposed = np.transpose(combined, (1,2,0,3))
	return transposed.reshape(HEIGHT, WIDTH, 6) # 6 channels

# Read in and preprocess an image pair, returning something we can feed into the network
def read_preprocess(left, right):
	im1 = cv2.imread(left)
	im2 = cv2.imread(right)

	im1 = cv2.resize(im1, (128, 128))
	im2 = cv2.resize(im2, (128, 128))

	return combine_images(im1, im2)

# Actually have the network predict whether the images are duplicates.
def predict(data):
	test_input_fn = tf.contrib.learn.io.numpy_input_fn(
		x = {"x": np.array([data])},
		num_epochs = 1,
		shuffle = False
	)

	# Do the prediction.
	test_results = estimator.predict(
		input_fn = test_input_fn,
	)

	return list(test_results)[0]['probabilities'][1]

# Read in two images, and return the confidance that the two are duplicates.
def get_confidance(left, right):
	# Initialize the estimator on the first time calling this function.
	if estimator is None:
		init_estimator()

	data = read_preprocess(left, right)

	return predict(data)

# Print out the confidance that the first two command-line arguments are equal.
def main(argv):
	print(get_confidance(argv[1], argv[2]))

import sys

if __name__ == "__main__":
	main(sys.argv)
