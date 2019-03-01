import numpy as np
import tensorflow as tf
from operations import *
import matplotlib.pyplot as plt # For generating images
import os # For creating output folder
from generator import Generator
from discriminator import Discriminator

class DCGAN:
    def __init__(self, img_shape, epochs=50000, lr_gen=0.0001, lr_dc=0.0001, z_shape=100, batch_size=64, beta1=0.5, epochs_for_sample=500):

        # lr_gen = Learning rate for Generator
        # lr_dc = Learning rate for Discriminator
        # z_shape = Shape for generator input
        # batch_size can be changed --> bigger = slower training/epochs--> smaller = faster training/epochs(but needs more epochs)
        # epochs_for_sample --> Interval for genrating images

        # Unpack image Shape
        self.rows, self.cols, self.channels = img_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.z_shape = z_shape
        self.epochs_for_sample = epochs_for_sample
        self.generator = Generator(img_shape, self.batch_size)
        self.discriminator = Discriminator(img_shape)

        # Load MNIST dataset
        mnist = tf.keras.datasets.mnist
        (x_train, _), (x_test, _) = mnist.load_data()

        # Labels not needed
        # Differentiation between x_train and x_test not needed --> Concat x_train and x_test
        X = np.concatenate([x_train, x_test])
        # Values between 0 and 255
        # Scale between -1 and 1
        self.X = X/127.5 - 1

        # Create placeholders for input
        self.phX = tf.placeholder(tf.float32, [None, self.rows, self.cols])
        self.phZ = tf.placeholder(tf.float32, [None, self.z_shape])

        # Generator forward pass
        self.gen_out = self.generator.forward(self.phZ)

        # Discriminator prediction
        dc_logits_fake = self.discriminator.forward(self.gen_out)

        # Real images
        dc_logits_real = self.discriminator.forward(self.phX)

        # Cost functions
        # Discriminator should predict that fake images are 0 and real images are 1

        dc_fake_loss = cost(tf.zeros_like(dc_logits_fake), dc_logits_fake)
        dc_real_loss = cost(tf.ones_like(dc_logits_real), dc_logits_real)

        self.dc_loss = tf.add(dc_fake_loss, dc_real_loss)

        # Generator tries to fool discriminator so that the discriminator outputs 1 for fake images
        self.gen_loss = cost(tf.ones_like(dc_logits_fake), dc_logits_fake)

        train_vars = tf.trainable_variables()

        # Differentiating between generator and discriminator variables
        dc_vars = [var for var in train_vars if 'd' in var.name]
        gen_vars = [var for var in train_vars if 'g' in var.name]

        # Create training variables
        self.dc_train = tf.train.AdamOptimizer(lr_dc, beta1= beta1).minimize(self.dc_loss, var_list=dc_vars)
        self.gen_train = tf.train.AdamOptimizer(lr_gen, beta1= beta1).minimize(self.gen_loss, var_list=gen_vars)


    def train(self):
        init = tf.global_variables_initializer()
        self.sess = tf.Session()

        # Initialize all variables
        self.sess.run(init)

        # Start training loop
        for i in range(self.epochs):
            # Create random batch for training
            # Create random indices (minimum: 0, maxmium: size of X, size: batch_size)
            idx = np.random.randint(0, len(self.X), self.batch_size)
            batch_X = self.X[idx]
            batch_Z = np.random.uniform(-1, 1, (self.batch_size, self.z_shape))

            # Train discriminator and store dc loss
            # batch_X = batch_X.reshape([-1, 28, 28, 1])
            _, d_loss = self.sess.run([self.dc_train, self.dc_loss], feed_dict={self.phX:batch_X, self.phZ:batch_Z})

            # Create new batch for generator training
            batch_Z = np.random.uniform(-1, 1, (self.batch_size, self.z_shape))

            # Train generator and store generator loss
            _, g_loss = self.sess.run([self.gen_train, self.gen_loss], feed_dict={self.phZ:batch_Z})

            # Generate samples and print loss
            if i % self.epochs_for_sample == 0:
                self.generate_sample(i)
                print(f"Epoch: {i}. Discriminator loss: {d_loss}. Generator loss: {g_loss}")

    def generate_sample(self, epoch):
        # 7 sample per image
        c = 7
        r = 7

        # New input for sample
        # 7x7 = 49 image samples
        z = np.random.uniform(-1, 1, (self.batch_size, self.z_shape))
        imgs = self.sess.run(self.gen_out, feed_dict={self.phZ:z})

        # Scale back to values between 0 and 1 (currently between -1 and 1)
        imgs = imgs*0.5 + 0.5

        # Create subplots
        fig, axs = plt.subplots(c, r)
        count = 0
        for i in range(c):
            for j in range(r):
                axs[i, j].imshow(imgs[count, :, :, 0], cmap="gray")
                axs[i, j].axis("off")
                count += 1

        # Save images
        fig.savefig("DCGAN 01/samples/%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    img_shape = (28, 28, 1)
    epochs = 50000
    dcgan = DCGAN (img_shape, epochs)

    # Create sample folder
    if not os.path.exists("DCGAN 01/samples/"):
        os.makedirs("DCGAN 01/samples/")

    # self.SAMPLE_FOLDER_NAME = sample_folder_name

    dcgan.train()
