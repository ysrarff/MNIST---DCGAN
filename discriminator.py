import numpy as np
import tensorflow as tf
from operations import *
from tensorflow.layers import batch_normalization

class Discriminator:
    def __init__(self, img_shape):
        _, _, channels = img_shape
        # Initializing all weights and bias
        # Variable scope is needed so that generator and discriminator can be differentiated
        self.img_rows, self.img_cols, self.channels = img_shape
        with tf.variable_scope('d'):
            print("Initializing discriminator weights")
            self.W1 = init_weights([5, 5, channels, 64])
            self.b1 = init_bias([64])
            self.W2 = init_weights([3, 3, 64, 64])
            self.b2 = init_bias([64])
            self.W3 = init_weights([3, 3, 64, 128])
            self.b3 = init_bias([128])
            self.W4 = init_weights([2, 2, 128, 256])
            self.b4 = init_bias([256])
            self.W5 = init_weights([7*7*256, 1])
            self.b5 = init_bias([1])

    def forward(self, X, momentum=0.5):
        # Create forward pass
        # 4 conv2d layers -->No use of pooling --> we use stride of size 2 instead to decrease image size
        # 1 fully connected layer

        # Strides shape: [batch, height, width, channels]
        X = tf.reshape(X, [-1, self.img_rows, self.img_cols, self.channels])
        z = conv2d(X, self.W1, [1, 2, 2, 1], padding="SAME") # Size = 14 x 14 x 64

        #Add bias
        z = tf.nn.bias_add(z, self.b1)

        # Activation function: leaky relu
        z = tf.nn.leaky_relu(z)

        # 2nd layer
        z = conv2d(z, self.W2, [1, 1, 1, 1], padding="SAME") # Size = 14 x 14 x 64
        z = tf.nn.bias_add(z, self.b2)
        # Batch normalization with momentum = 0.5
        z = batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)

        # 3rd layer
        z = conv2d(z, self.W3, [1, 2, 2, 1], padding="SAME") # Size = 7 x 7 x 128
        z = tf.nn.bias_add(z, self.b3)
        # Batch normalization with momentum = 0.5
        z = batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)

        # 4th layer
        z = conv2d(z, self.W4, [1, 1, 1, 1], padding="SAME") # Size = 7 x 7 x 26
        z = tf.nn.bias_add(z, self.b4)
        # Batch normalization with momentum = 0.5
        z = batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)

        # Fully connected output layer
        # Flatten image
        z = tf.reshape(z, [-1, 7*7*256])
        logits = tf.matmul(z, self.W5)
        logits = tf.nn.bias_add(logits, self.b5)
        # No activation function needed because tensorflow includes it in cost function
        return logits
