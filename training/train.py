from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Library imports
import numpy as np
import tensorflow as tf
import cv2 #cv2.imread for reading images.
import os #path

# Local file imports
from model_fn import cnn_model_fn
from config import * #Configuration options like HEGIHT or WIDTH

tf.logging.set_verbosity(tf.logging.INFO)

# Combines the two imported images into one numpy.ndarray with 6 "channels"
def combine_images(im1, im2):
	combined = np.array((im1, im2), dtype=np.float16)
	transposed = np.transpose(combined, (1,2,0,3))
	return transposed.reshape(HEIGHT, WIDTH, 6) # 6 channels

# Returns a tuple containing (data, labels), where data is a 4-d array and labels is a 1D array.
# kind should be train, test, or validation
def get_data(kind):
	# Path to the data
	testpath = os.path.join('..', 'data', 'output', kind)
	# Paths to the true and false test cases
	truepath = os.path.join(testpath, 'true_cases')
	falsepath = os.path.join(testpath, 'false_cases')

	# The data we will return.
	data = []
	labels = [] #The labels we will return.
	files = sorted(os.listdir(truepath))
	for f1,f2 in zip(files[::2], files[1::2]):
		# Go through pairs of files.
		im1 = cv2.imread(f1)
		im2 = cv2.imread(f2)
		
		data.append(combine_images(im1, im2))
		labels.append(true) #This is a true case.

	# Now import the false test cases.
	files = sorted(os.listdir(falsepath))
	for f1,f2 in zip(files[::2], files[1::2]):
		# Go through pairs of files.
		im1 = cv2.imread(f1)
		im2 = cv2.imread(f2)
		
		data.append(combine_images(im1, im2))
		labels.append(false) #This is a true case.

	return np.array(data), np.array(labels)

def main(unused_argv):
	# Load the training and test data.
	train_data, train_labels = get_data('train') # Returns an np.array
	test_data, test_labels = get_data('test') # Returns an np.array

	#Allocate 90% of the GPU's memory.
	config = tf.contrib.learn.RunConfig(gpu_memory_fraction=0.9)
	
	mnist_classifier = tf.contrib.learn.Estimator(
		model_fn = cnn_model_fn,
		model_dir = os.path.join("..", 'data', 'generated_network'),
		config = config
		)

	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=500)
	
#	train_input_fn = tf.estimator.inputs.numpy_input_fn(
	train_input_fn = tf.contrib.learn.io.numpy_input_fn(
		x = {"x": train_data},
		y = train_labels,
		batch_size = BATCH_SIZE,
		num_epochs = EPOCHS,
		shuffle = True
	)
	

	mnist_classifier.fit(
		input_fn = train_input_fn,
#		max_steps = 20000,
		monitors = [logging_hook]
		)
	
#	mnist_classifier.train(
#		input_fn = train_input_fn,
#		steps = 20000,
#		hooks = [logging_hook]
#	)

#	test_input_fn = tf.estimator.inputs.numpy_input_fn(
	test_input_fn = tf.contrib.learn.io.numpy_input_fn(
		x = {"x": test_data},
		y = test_labels,
		num_epochs = 1,
		shuffle = False
	)
	
	test_results = mnist_classifier.evaluate(
		input_fn = test_input_fn
	)
	

	print(test_results)

if __name__ == "__main__":
  tf.app.run()
  


