from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

from config import * #Configuration options, like HEIGHT or WIDTH.

def leaky_relu(x, alpha):
	return tf.maximum(x, alpha*x)

# Features is the input data, in a batch.
# Labels is a 1D array of result vaules.
def cnn_model_fn(features, labels, mode):
#	input_layer = tf.reshape(features["x"], [-1, HEIGHT, WIDTH, 6])

	input_layer = tf.cast(features["x"], tf.float32)/256

	""" 
	conv1 = tf.layers.conv2d(
		inputs = input_layer,
		filters = CHANNELS[0],
		kernel_size = [KERNEL_SIZE,KERNEL_SIZE],
		#Padding = same means that output is same dimension as input (last is 32)
		padding = "same") #,
#		activation = tf.nn.relu)
		
	conv1 = tf.layers.batch_normalization(leaky_relu(conv1,0.01))
	
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[POOL_SIZE,POOL_SIZE], strides = 2)

	conv2 = tf.layers.conv2d(
		inputs = pool1,
		filters = CHANNELS[1],
		kernel_size = [KERNEL_SIZE, KERNEL_SIZE],
		#Padding = same means that output is same dimension as input (last is 32)
		padding = "same",
		activation = tf.nn.relu)

	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)

	pool_flat = tf.reshape(pool2, [-1, HEIGHT//2**LAYERS * WIDTH//2**LAYERS * CHANNELS[LAYERS-1]])
	"""

	previous = input_layer
	for i in range(LAYERS):
		print(i)
		conv = tf.layers.conv2d(
			inputs = previous,
			filters = CHANNELS[i],
			kernel_size = [KERNEL_SIZE,KERNEL_SIZE],
			#Padding = same means that output is same dimension as input (last is 32)
			padding = "same") #,
#			activation = tf.nn.relu)
		
		conv = tf.layers.batch_normalization(leaky_relu(conv,0.01))
		
		pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[POOL_SIZE,POOL_SIZE], strides = 2)

		previous = pool


	pool_flat = tf.reshape(previous, [-1, HEIGHT//2**LAYERS * WIDTH//2**LAYERS * CHANNELS[LAYERS-1]])
	
	dense = tf.layers.dense(inputs=pool_flat, units=DENSE_NODES) #, activation=tf.nn.relu)
	
	#Apply leaky relu activation function.
	dense = leaky_relu(dense, 0.01)

	dropout = tf.layers.dropout(
		inputs = dense,
		rate = 0.4,
		training = mode == tf.contrib.learn.ModeKeys.TRAIN
	)
	
	# This is a binary classifier, so only 1 output node.
	logits = tf.layers.dense(inputs=dropout, units=2)

	predictions = {
		"classes" : tf.argmax(input=logits, axis=1, name="classes"),
		"probabilities" : tf.nn.softmax(logits)
	}
	
#	if mode == tf.estimator.ModeKeys.PREDICT:
	if mode == tf.contrib.learn.ModeKeys.INFER:
		return tf.contrib.learn.ModelFnOps(mode=mode, predictions=predictions)

	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
	
	# Log Loss is good for binary classifiers.
#	loss = tf.losses.log_loss(labels=labels, predictions=logits)
#	loss = tf.losses.absolute_difference(labels=labels, predictions=logits)
	loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

	if mode == tf.contrib.learn.ModeKeys.TRAIN:
		# Create optimizer object with learning rate.
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
#		optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
		train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
		
		return tf.contrib.learn.ModelFnOps(mode=mode, loss=loss, train_op=train_op)
	
	# In EVAL mode
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
	}
	
#	return tf.estimator.ModelFnOps(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
	return tf.contrib.learn.ModelFnOps(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, predictions=predictions)

