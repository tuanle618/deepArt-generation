# -*- coding: utf-8 -*-
"""
@title train_model.py
@author: Tuan Le
@email: tuanle@hotmail.de
This script builds a generative adversarial network using keras framework with tensorflow backend. 
In this case the deep convolutional generative adversarial network will be used as model, as we are dealing
with colour images and suggested DCGAN work better than the classical GAN.
Proposed papers:
    - original: https://arxiv.org/pdf/1511.06434.pdf
    - additional: https://arxiv.org/pdf/1406.2661.pdf
"""

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

## Load images from preprocess script ##
from data_preprocess import preprocess
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

## Build class for DCGAN

class DCGAN():
    def __init__(self):
        # Input image shape
        self.rows = 256
        self.cols = 256
        self.channels = 3
        self.img_shape = (self.rows, self.cols, self.channels)
        # Latent vector Z with dimension 100
        self.latent_dim = 100
        optimizer = Adam(lr = 0.0002, beta_1 = 0.5, beta_2  = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
        
        ## Discriminator:
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss = "binary_crossentropy",
                                   optimizer = optimizer,
                                   metrics = ["accuracy"])
        
        ## Generator:
        self.generator = self.build_generator()
        ## The generator gets as input noise sampled from the latent space vector Z
        Z = Input(shape = (self.latent_dim, ))
        ## generate image with noisy latent input vector Z
        generated_img = self.generator(Z)
        
        ## The discriminator takes the generated image as input and either classifies them as real or fake
        discrim_classify = self.discriminator(generated_img)
        
        ## Combine generator and discriminator to simultaneously optimize the weights as a stacked (adversarial) model overall
        self.stacked_G_D = Model(inputs = Z, outputs = discrim_classify)
        ## In DCGAN only the generator will be trained to create images look-alike the "real" images to fool the discriminator: freeze D-weights
        self.discriminator.trainable = False
        self.stacked_G_D.compile(loss = "binary_crossentropy", optimizer = optimizer)
        
        
        ## Class functions to build discriminator and generator (networks):
        
        ## Discriminator network
    def build_discriminator(self):            
        discrim_model = Sequential()
        
        #Layer 1:
        #Input: 256x256x3
        #Output: 128x128x32
        discrim_model.add(Conv2D(filters = 32, kernel_size = 3, strides = 2, input_shape = self.img_shape, padding = "same"))
        discrim_model.add(LeakyReLU(alpha = 0.2))
        discrim_model.add(Dropout(rate = 0.25))
        #Layer 2:
        #Input: 128x128x32
        #Output: 64x64x64
        discrim_model.add(Conv2D(filters = 64, kernel_size = 3, strides = 2, padding = "same"))
        discrim_model.add(BatchNormalization(momentum = 0.8))
        discrim_model.add(LeakyReLU(alpha = 0.2))
        discrim_model.add(Dropout(rate = 0.25))
        #Layer 3:
        #Input: 64x64x64
        #Output: 32x32x128
        discrim_model.add(Conv2D(filters = 128, kernel_size = 3, strides = 2, padding = "same"))
        discrim_model.add(BatchNormalization(momentum = 0.8))
        discrim_model.add(LeakyReLU(alpha = 0.2))
        discrim_model.add(Dropout(rate = 0.25))
        #Layer 4:
        #Input: 16x16x256
        #Output: 1-dimensional probability 
        discrim_model.add(Conv2D(filters = 256, kernel_size = 3, strides = 2, padding = "same"))
        discrim_model.add(BatchNormalization(momentum = 0.8))
        discrim_model.add(LeakyReLU(alpha = 0.2))
        discrim_model.add(Dropout(rate = 0.25))
        discrim_model.add(Flatten())
        discrim_model.add(Dense(1, activation = "sigmoid"))
            
        print("Architecture for discriminator network:")
        print(discrim_model.summary())
            
        ##Feed the discrimator with an image
            
        img = Input(shape = self.img_shape)
        classify_img = discrim_model(img)
            
        return Model(inputs = img, outputs = classify_img)
        
    
        ## Generator network
    def build_generator(self):
        gen_model = Sequential()
        
        #Layer 1:
        #Input: random noise = 100
        #Output: 64x64x128
        gen_model.add(Dense(units = 64*64*128, activation = "relu", input_dim = self.latent_dim))
        gen_model.add(Reshape((64, 64, 128)))
        
        #Layer 2
        #Input: 64x64x128
        #Output: 128x128x128
        gen_model.add(UpSampling2D())
        gen_model.add(Conv2D(filters = 128, kernel_size = 3, padding = "same"))
        gen_model.add(BatchNormalization(momentum = 0.8))
        gen_model.add(Activation("relu"))

        #Layer 3
        #Input: 128x128x128
        #Output: 256x256x64
        gen_model.add(UpSampling2D())
        gen_model.add(Conv2D(filters = 64, kernel_size = 3, padding = "same"))
        gen_model.add(BatchNormalization(momentum = 0.8))
        gen_model.add(Activation("relu"))
        
        #Layer 4
        #Input: 256x256x64
        #Output: 256x256x3
        gen_model.add(Conv2D(filters = self.channels, kernel_size = 3, padding = "same"))
        gen_model.add(Activation("tanh"))
        
        print("Architecture for generator network:")
        print(gen_model.summary())
        
        latent_noise = Input(shape=(self.latent_dim,))
        generated_img = gen_model(latent_noise)
        
        return Model(inputs = latent_noise, outputs = generated_img)
        
    
    ## Helper for scaling and unscaling:
    def scale(self, x, out_range = (-1, 1)):
        #domain = np.min(x), np.max(x)
        # a)scale data such that its symmetric around 0
        #y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
        # b)rescale data such that it falls into desired output range
        #y = y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
        
        ###new##
        y = 2 * (x - np.min(x)) / np.ptp(x) - 1
        return y
    
    def unscale(self, x, x_unscaled, out_range = (-1, 1)):
        #domain = np.min(x_unscaled), np.max(x_unscaled)
        # undo b)
        #y = ( x - (out_range[1] + out_range[0]) / 2 ) / (out_range[1] - out_range[0])
        # undo a)
        #y = y * (domain[1] - domain[0]) + (domain[1] + domain[0]) / 2
        
        ###new###
        y = (x + 1) * 0.5 * np.ptp(x_unscaled) + np.min(x_unscaled)
        return y

    
    def train(self, data, epochs = 100, batch_size = 10, save_intervals = 20):
        #since tanh is used as activation for generator in the very last layer, need to rescale real input images into range [-1,1]
        final_images_stacked = data
        X_train = self.scale(x = data) 

        #adversarial truth:
        valid = np.ones(shape = (batch_size, 1))
        fake = np.zeros(shape = (batch_size, 1))
        
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Select a random half of ground-truth images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new fake images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated fake as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            # Concatenate total loss from real and fake into mean vector
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
            # ---------------------
            #  Train Generator
            # ---------------------

            # Indirectly train the generator through the adversarial (stacked) model:
                # Pass noise to the adversarial model and mislabel everything as if they were images taken from the true database,
                # When they will be generated by the generator.
            g_loss = self.stacked_G_D.train_on_batch(noise, valid)
    
            # Print the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
    
            # If at save interval => save generated image samples + model weights
            if epoch % save_intervals == 0:
                self.save_imgs(epoch = epoch, final_images_stacked = final_images_stacked)
                self.generator.save_weights(filepath = "../model/epoch_" + str(epoch) + "_generator.h5")
                self.discriminator.save_weights(filepath = "../model/epoch_" + str(epoch) + "_discriminator.h5")
            
            #If last epoch generate 10 images in full mode:
            if epoch == (epochs - 1):
                final_noises = np.random.normal(0, 1, (10, self.latent_dim))
                final_gen_images = self.generator.predict(final_noises)
                final_gen_images = self.unscale(x = final_gen_images, x_unscaled = final_images_stacked)
                for i in range(10):
                    plt.imshow(final_gen_images[i, :, :, :], interpolation = "nearest")
                    plt.savefig("../model/images/final_generated_images_%d.jpg" % i)
                    
        
        
    def save_imgs(self, epoch, final_images_stacked):
        r, c = 2, 2
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
    
    # Rescale images 0 - 1
        gen_imgs = self.unscale(x = gen_imgs, x_unscaled = final_images_stacked)
    
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis("off")
                cnt += 1
        fig.savefig("../model/images/dcgan_%d.jpg" % epoch)
        plt.close(fig)
 

if __name__ == "__main__":
    print("Python main programm for generating images using DCGAN")
    #flag for training or after training execution
    do_train = True
    
    my_dcgan = DCGAN()
    
    ## for docu ##
    #from keras.utils import plot_model
    #os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    #plot_model(my_dcgan.generator, to_file = "../model/generator_model.png", show_shapes = True)
    #plot_model(my_dcgan.discriminator, to_file = "../model/discriminator_model.png", show_shapes = True)
    ##############
    
    res = preprocess()
    final_images_stacked = res[1]
    import os
    os.chdir("..")
    os.chdir("..")
    os.chdir("source")
    if do_train:
        print("Start training the DCGAN:")
        my_dcgan.train(data = final_images_stacked, epochs = 4600, batch_size = 10, save_intervals = 200)
    else:
        print("Using most updated weights of generator and discriminator for the stacked (adversarial) model:")
        generator_weights = "../model/epoch_4400_generator.h5"
        discrimininator_weights = "../model/epoch_4400_discriminator.h5"
        #load generator weights
        my_dcgan.generator.load_weights(filepath = generator_weights)
        my_dcgan.discriminator.load_weights(filepath = discrimininator_weights)
        final_noises = np.random.normal(0, 1, (10, my_dcgan.latent_dim))
        final_gen_images = my_dcgan.generator.predict(final_noises)
        final_gen_images = my_dcgan.unscale(x = final_gen_images, x_unscaled = final_images_stacked)
        for i in range(10):
            plt.imshow(final_gen_images[i], interpolation = "nearest")
            plt.axis("off")
            plt.imsave("../model/images/final_generated_images_%d.jpg" % i, final_gen_images[i])
            plt.show()
            plt.close()
            