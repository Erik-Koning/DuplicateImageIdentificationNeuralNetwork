from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

from config import * #Configuration options, like HEIGHT or WIDTH.

# Tensorflow < 1.4 compatibility.
tf.data.Dataset = tf.contrib.data.Dataset


def decompress_jpeg(image):
	return tf.image.decode_jpeg(image, channels=3)

# Combine 2 images of 3 channels each into one image with 6 channels.
def combine_images(im1, im2):
	# axis=2 => concat the channels.
	return tf.concat((im1, im2), 2)

# Images is 1D array containing 2 images as binary strings.
def decompress_combine(images):
	im1 = decompress_jpeg(images[0])
	im2 = decompress_jpeg(images[0])
	return combine_images(im1, im2)

# Similar to numpy input function
# x is a dict of features, each feature being a numpy array.
# y is a numpy array of labels.
def make_input_fn(x, y, batch_size = 128, epochs = 1, shuffle = True):
	
	# Convert all features from numpy arrays to tensorflow tensors.
	for k in x:
		x[k] = tf.convert_to_tensor(x[k])
	
	dataset = tf.data.Dataset.from_tensor_slices(x,y)
	
	# Max items to shuffle as they pass through.
	SHUFFLE_BUFFER_SIZE = 1000
	
	# Apply transformations
	if shuffle:
		dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)
	
	# Do the speicifed number of epochs.
	dataset.repeat(epochs)
	
	# Batch the input.
	dataset = dataset.batch(batch_size)
	
	# Decompress the jpegs, and combine each pair of images into one input.
	dataset = dataset.map(decompress_combine)

	return dataset
