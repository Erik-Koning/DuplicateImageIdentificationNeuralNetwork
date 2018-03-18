from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Library imports
import numpy as np
import tensorflow as tf

# Local file imports
from model_fn import cnn_model_fn

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):
	# Load the training and test data.
	mnist = tf.contrib.learn.datasets.load_dataset("mnist")
	train_data = mnist.train.images # Returns an np.array
	train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
	test_data = mnist.test.images
	test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

	#Allocate 90% of the GPU's memory.
	config = tf.contrib.learn.RunConfig(gpu_memory_fraction=0.9)
	
	mnist_classifier = tf.contrib.learn.Estimator(
		model_fn = cnn_model_fn,
		model_dir = "/tmp/mnist_convnet_model",
		config = config
		)

	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=500)
	
#	train_input_fn = tf.estimator.inputs.numpy_input_fn(
	train_input_fn = tf.contrib.learn.io.numpy_input_fn(
		x = {"x": train_data},
		y = train_labels,
		batch_size = 100,
		num_epochs = None,
		shuffle = True
	)
	

	mnist_classifier.fit(
		input_fn = train_input_fn,
		max_steps = 20000,
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
  


