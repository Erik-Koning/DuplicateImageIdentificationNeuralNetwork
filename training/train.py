#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Library imports
import numpy as np
import tensorflow as tf
import os #path

# Local file imports
from model_fn import cnn_model_fn
from config import * #Configuration options like HEGIHT or WIDTH
import read_data #For reading in the data

tf.logging.set_verbosity(tf.logging.INFO)

def train(classifier, train_input_fn):
	classifier.fit(
		input_fn = train_input_fn,
#		max_steps = 20000,
#		monitors = [logging_hook]
	)
	

def main(argv):
	# -1 = iterate until user stops it.
	iterations = -1
	if len(argv) > 1:
		# Can provide iteration count.
		iterations = int(argv[1])
	
	#Allocate 90% of the GPU's memory.
	config = tf.contrib.learn.RunConfig(gpu_memory_fraction=0.9)
	
	mnist_classifier = tf.contrib.learn.Estimator(
		model_fn = cnn_model_fn,
		model_dir = os.path.join("..", 'data', 'generated_network'),
		config = config
		)

	if DO_TRAIN:
		# Load the training data.
		print("Importing training data...")
		train_data, train_labels = read_data.get_data('train')
		print("Data importing has finished.")
	
		tensors_to_log = {"classes": "classes"}
		logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
		
#		train_input_fn = tf.estimator.inputs.numpy_input_fn(
		train_input_fn = tf.contrib.learn.io.numpy_input_fn(
			x = {"x": train_data},
			y = train_labels,
			batch_size = BATCH_SIZE,
			num_epochs = EPOCHS,
			shuffle = True
		)
		
		
		
#		mnist_classifier.train(
#			input_fn = train_input_fn,
#			steps = 20000,
#			hooks = [logging_hook]
#		)

	
	if DO_TEST:
	
		print("Importing testing data...")
		test_data, test_labels = read_data.get_data('test')
		print("Data importing has finished.")

#		test_input_fn = tf.estimator.inputs.numpy_input_fn(
		test_input_fn = tf.contrib.learn.io.numpy_input_fn(
			x = {"x": test_data},
			y = test_labels,
			num_epochs = 1,
			shuffle = False
		)
	
		
	# Python 2/3 compatibility.
	input = raw_input

	while (True):
		if DO_TRAIN:
			train(mnist_classifier, train_input_fn)

		if DO_TEST:
			test(mnist_classifier, test_input_fn)

		iterations -= 1
		# If we're out of iterations, ask the user if
		# they want to continue.
		if iterations <= 0:
			choice = input("Continue? [Y/n]:")
			if choice.lower()=='n':
				break
			choice = input("How many more?:")
			try:
				interations = int(choice)
			except ValueError as e:
				# Just do another interation
				pass


	# If we're testing, log the classes.
	if DO_TEST:
		test_results = mnist_classifier.predict(
			input_fn = test_input_fn,
		)
	
		# Place to put the comparison between predicted and actual class.
		output_path = os.path.join("..", "data", "class_comparison.txt")
		predictions = [res['probabilities'] for res in test_results]
		with open(output_path, "w") as fp:
			print("Left: predicted, Right: actual", file=fp)
			for pred, actual in zip(predictions, test_labels):
				print(pred, actual, file=fp)


def test(classifier, test_input_fn,
         output_path = os.path.join("..", "data", "results.txt")
        ):

	test_results = classifier.evaluate(
		input_fn = test_input_fn,
	)

	#Append the results to the results file.
	with open(output_path, "a") as fp:
		print(test_results, file=fp)


if __name__ == "__main__":
  tf.app.run()
  


