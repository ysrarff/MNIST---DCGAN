# Collection of operations that are needed

import numpy as np
import tensorflow as tf

# Help function for weights
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape=shape, stddev=0.02)) # Std Dev can be changed

# Help function for bias
def init_bias(shape):
    return tf.Variable(tf.zeros(shape))

# Help function for creating convolutional layer
def conv2d(x, filter, strides, padding):
    return tf.nn.conv2d(x, filter, strides=strides, padding=padding)

# Help function for creating cost function
def cost(labels, logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

# Sigmoid -> Creates output between 0 and 1 == Fake or Real Image
