from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.estimator = tf.contrib.learn

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

# Features is the input data, in a batch.
# Labels is a 1D array of result vaules.
def cnn_model_fn(features, labels, mode):
	# Input Layer
	input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
#	input_layer = tf.reshape(features, [-1, 28, 28, 1])

	me_too_thanks = "same"

	conv1 = tf.layers.conv2d(
		inputs = input_layer,
		filters = 32,
		kernel_size = [5,5],
		#Padding = same means that output is same dimension as input (last is 32)
		padding = me_too_thanks,
		activation = tf.nn.relu)
	
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides = 2)

	conv2 = tf.layers.conv2d(
		inputs = pool1,
		filters = 64,
		kernel_size = [5,5],
		#Padding = same means that output is same dimension as input (last is 32)
		padding = me_too_thanks,
		activation = tf.nn.relu)

	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)
	
	pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
	
	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

	dropout = tf.layers.dropout(
		inputs = dense,
		rate = 0.4,
		training = mode == tf.estimator.ModeKeys.TRAIN
	)
	
	logits = tf.layers.dense(inputs=dropout, units=10)
	
	predictions = {
		"classes" : tf.argmax(input=logits, axis=1),
		"probabilities" : tf.nn.softmax(logits, name="softmax_tensor")
	}
	
#	if mode == tf.estimator.ModeKeys.PREDICT:
	if mode == tf.estimator.ModeKeys.INFER:
		return tf.estimator.ModelFnOps(mode=mode, predictions=predictions)
	
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)

	loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)


	if mode == tf.estimator.ModeKeys.TRAIN:
		# Create optimizer object with learning rate.
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
		
		return tf.estimator.ModelFnOps(mode=mode, loss=loss, train_op=train_op)
	
	# In EVAL mode
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
	}
	
#	return tf.estimator.ModelFnOps(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
	return tf.estimator.ModelFnOps(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, predictions=predictions)

def main(unused_argv):
	# Load the training and eval data.
	mnist = tf.contrib.learn.datasets.load_dataset("mnist")
	train_data = mnist.train.images # Returns an np.array
	train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
	eval_data = mnist.test.images
	eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

	#Allocate 90% of the GPU's memory.
	config = tf.contrib.learn.RunConfig(gpu_memory_fraction=0.9)
	
	mnist_classifier = tf.estimator.Estimator(
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
#		steps = 1500,
#		hooks = [logging_hook]
#	)

#	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	eval_input_fn = tf.contrib.learn.io.numpy_input_fn(
		x = {"x": eval_data},
		y = eval_labels,
		num_epochs = 1,
		shuffle = False
	)
	
	eval_results = mnist_classifier.evaluate(
		input_fn = eval_input_fn
	)
	

	print(eval_results)

if __name__ == "__main__":
  tf.app.run()
  


