from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

from config import * #Configuration options, like HEIGHT or WIDTH.

# Features is the input data, in a batch.
# Labels is a 1D array of result vaules.
def cnn_model_fn(features, labels, mode):
	# Input Layer, combine the two images into one image with 6 channels.
	input_layer = tf.reshape(features["x"], [-1, HEIGHT, WIDTH, 6])

	conv1 = tf.layers.conv2d(
		inputs = input_layer,
		filters = CHANNELS[0],
		kernel_size = [5,5],
		#Padding = same means that output is same dimension as input (last is 32)
		padding = "same",
		activation = tf.nn.relu)
	
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides = 2)

	conv2 = tf.layers.conv2d(
		inputs = pool1,
		filters = CHANNELS[2],
		kernel_size = [5,5],
		#Padding = same means that output is same dimension as input (last is 32)
		padding = "same",
		activation = tf.nn.relu)

	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)
	
	pool2_flat = tf.reshape(pool2, [-1, HEIGHT/2**LAYERS * WIDTH/2**LAYERS * CHANNELS[2]])
	
	dense = tf.layers.dense(inputs=pool2_flat, units=DENSE_NODES, activation=tf.nn.relu)

	dropout = tf.layers.dropout(
		inputs = dense,
		rate = 0.4,
		training = mode == tf.contrib.learn.ModeKeys.TRAIN
	)
	
	# This is a binary classifier, so only 1 output node.
	logits = tf.layers.dense(inputs=dropout, units=1, activation=tf.nn.sigmoid)
	
	predictions = {
		"classes" : tf.argmax(input=logits, axis=1),
		"probabilities" : logits
	}
	
#	if mode == tf.estimator.ModeKeys.PREDICT:
	if mode == tf.contrib.learn.ModeKeys.INFER:
		return tf.contrib.learn.ModelFnOps(mode=mode, predictions=predictions)
	
	# Log Loss is good for binary classifiers.
	loss = tf.losses.log_loss(labels=labels, predictions=logits)

	if mode == tf.contrib.learn.ModeKeys.TRAIN:
		# Create optimizer object with learning rate.
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
		
		return tf.contrib.learn.ModelFnOps(mode=mode, loss=loss, train_op=train_op)
	
	# In EVAL mode
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
	}
	
#	return tf.estimator.ModelFnOps(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
	return tf.contrib.learn.ModelFnOps(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, predictions=predictions)

