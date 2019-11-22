"""
@title dcgan.py
@author: Tuan Le
@email: tuanle@hotmail.de

This script builds a generative adversarial network using keras framework with tensorflow backend.
In this case the deep convolutional generative adversarial network will be used as model, as we are dealing
with colour images and suggested DCGAN work better than the classical GAN.
Proposed papers:
    - original: https://arxiv.org/pdf/1511.06434.pdf
    - additional: https://arxiv.org/pdf/1406.2661.pdf
This scripts includes more complex (deeper) Generator & Discriminator models.
"""

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, Adadelta

import numpy as np

#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
## Build class for DCGAN

class DCGAN():
    def __init__(self, name='DCGAN_1'):
        assert any(name.upper() in item for item in ['DCGAN_1', 'DCGAN_2', 'DCGAN_3']), 'Inserted <name>: "{}" is not provided in the list [DCGAN_1, DCGAN_2, DCGAN_3]'.format(name)
        self.name = name.upper()
        if not os.path.exists('../model/{}'.format(self.name)):
            os.makedirs('../model/{}'.format(self.name))
            os.makedirs('../model/{}/images'.format(self.name))
        # Input image shape
        self.rows = 128
        self.cols = 128
        self.channels = 3
        self.img_shape = (self.rows, self.cols, self.channels)
        # Number of filters for discriminator and generator network
        self.nf1 = 32
        self.nf2 = 64
        self.nf3 = 128
        self.nf4 = 256
        # Latent vector Z with dimension 100
        self.latent_dim = 100
        if self.name == 'DCGAN_1': 
            self.optimizer = Adam(lr = 0.0002, beta_1 = 0.5, beta_2  = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
            ## Build Discriminator and discriminator:
            self.discriminator = self.build_discriminator_1()
            self.discriminator.compile(loss = "binary_crossentropy",
                                       optimizer = self.optimizer,
                                       metrics = ["accuracy"])
            self.generator = self.build_generator_1()
        elif self.name == 'DCGAN_2':
            self.optimizer = Adadelta()
            ## Build Discriminator and discriminator:
            self.discriminator = self.build_discriminator_2()
            self.discriminator.compile(loss = "binary_crossentropy",
                                       optimizer = self.optimizer,
                                       metrics = ["accuracy"])
            self.generator = self.build_generator_2()
        elif self.name == 'DCGAN_3':
            self.optimizer = Adam()
            ## Build Discriminator and discriminator:
            self.discriminator = self.build_discriminator_3()
            self.discriminator.compile(loss = "binary_crossentropy",
                                       optimizer = self.optimizer,
                                       metrics = ["accuracy"])
            self.generator = self.build_generator_3()
        else:
            print('Model not available')
            
        #Build stacked DCGAN
        self.stacked_G_D = self.build_dcgan()
        
        
        ## Class functions to build discriminator and generator (networks):
        ########################## Architectures regarding the DCGAN ##########################
        ###################################### Model 1 ######################################
        
        ## Discriminator network
    def build_discriminator_1(self):
        discrim_model = Sequential()
        #Layer 1:
        #Input: 128x128x3
        #Output: 64x64x32
        discrim_model.add(Conv2D(filters = self.nf1, kernel_size = (3, 3), strides = (2, 2), input_shape = self.img_shape, padding = "same"))
        discrim_model.add(LeakyReLU(alpha = 0.2))
        discrim_model.add(Dropout(rate = 0.25))
        #Layer 2:
        #Input: 64x64x32
        #Output: 32x32x64
        discrim_model.add(Conv2D(filters = self.nf2, kernel_size = (3, 3), strides = (2, 2), padding = "same"))
        discrim_model.add(BatchNormalization(momentum = 0.8))
        discrim_model.add(LeakyReLU(alpha = 0.2))
        discrim_model.add(Dropout(rate = 0.25))
        #Layer 3:
        #Input: 32x32x64
        #Output: 16x16x128
        discrim_model.add(Conv2D(filters = self.nf3, kernel_size = (3, 3), strides = (2, 2), padding = "same"))
        discrim_model.add(BatchNormalization(momentum = 0.8))
        discrim_model.add(LeakyReLU(alpha = 0.2))
        discrim_model.add(Dropout(rate = 0.25))
        #Layer 4:
        #Input: 16x16x128
        #Output: 8x8x256
        discrim_model.add(Conv2D(filters = 256, kernel_size = 3, strides = 2, padding = "same"))
        discrim_model.add(BatchNormalization(momentum = 0.8))
        discrim_model.add(LeakyReLU(alpha = 0.2))
        discrim_model.add(Dropout(rate = 0.25))
        
        #Output Layer: 
        #Input: (8*8*256, )
        #Output: 1-dimensional probability
        discrim_model.add(Flatten())
        discrim_model.add(Dense(1, activation = "sigmoid", name='classif_discrim'))

        print("Architecture for discriminator network from model {}:".format(self.name))
        print(discrim_model.summary())

        ##Feed the discrimator with an image
        img = Input(shape = self.img_shape)
        classify_img = discrim_model(img)

        return Model(inputs = img, outputs = classify_img)

        ## Generator network
    def build_generator_1(self):
        gen_model = Sequential()

        #Layer 1:
        #Input: random noise = 100
        #Output: 8x8x256
        gen_model.add(Dense(units = 8*8*self.nf4, activation = "relu", input_dim = self.latent_dim, name='latent_dim_sample'))
        gen_model.add(Reshape((8,8, self.nf4)))

        #Layer 2
        #Input:  8x8x256
        #Output: 16x16x128
        gen_model.add(UpSampling2D())
        gen_model.add(Conv2D(filters = self.nf3, kernel_size = (3,3), strides=(1, 1), padding = "same"))
        gen_model.add(BatchNormalization(momentum = 0.8))
        gen_model.add(Activation("relu"))

        #Layer 3
        #Input: 16x16x128
        #Output: 32x32x64
        gen_model.add(UpSampling2D())
        gen_model.add(Conv2D(filters = self.nf2, kernel_size = (3,3), strides=(1, 1), padding = "same"))
        gen_model.add(BatchNormalization(momentum = 0.8))
        gen_model.add(Activation("relu"))
        
        #Layer 4
        #Input: 32x32x64
        #Output: 64x64x32
        gen_model.add(UpSampling2D())
        gen_model.add(Conv2D(filters = self.nf1, kernel_size = (3,3), strides=(1, 1), padding = "same"))
        gen_model.add(BatchNormalization(momentum = 0.8))
        gen_model.add(Activation("relu"))

        #Layer 5
        #Input: 64x64x32
        #Output: 128x128x32
        gen_model.add(UpSampling2D())
        gen_model.add(Conv2D(filters = self.nf1, kernel_size = (3,3), strides=(1, 1), padding = "same"))
        gen_model.add(BatchNormalization(momentum = 0.8))
        gen_model.add(Activation("relu"))
        
        #Output Layer:
        #Input: 128x128x32
        #Output: 128x128x3
        gen_model.add(Conv2D(filters = self.channels, kernel_size = 3, padding = "same"))
        gen_model.add(Activation("tanh", name='generated_image'))

        print("Architecture for generator network:")
        print(gen_model.summary())

        latent_noise = Input(shape=(self.latent_dim,))
        generated_img = gen_model(latent_noise)

        return Model(inputs = latent_noise, outputs = generated_img)
    #####################################################################################
    
    ###################################### Model 2 ######################################
   ## Discriminator network
    def build_discriminator_2(self):
        discrim_model = Sequential()
        #Layer 1:
        #Input: 128x128x3
        #Output: 64x64x32
        discrim_model.add(Conv2D(filters = self.nf1, kernel_size = (3, 3), strides = (1, 1), input_shape = self.img_shape, padding = "same"))
        discrim_model.add(MaxPooling2D(pool_size=(2, 2)))
        discrim_model.add(LeakyReLU(alpha = 0.05))
        discrim_model.add(Dropout(rate = 0.10))
        #Layer 2:
        #Input: 64x64x32
        #Output: 32x32x64
        discrim_model.add(Conv2D(filters = self.nf2, kernel_size = (3, 3), strides = (1, 1), padding = "same"))
        discrim_model.add(MaxPooling2D(pool_size=(2, 2)))
        discrim_model.add(BatchNormalization())
        discrim_model.add(LeakyReLU(alpha = 0.05))
        discrim_model.add(Dropout(rate = 0.10))
        #Layer 3:
        #Input: 32x32x64
        #Output: 16x16x128
        discrim_model.add(Conv2D(filters = self.nf3, kernel_size = (3, 3), strides = (1, 1), padding = "same"))
        discrim_model.add(MaxPooling2D(pool_size=(2, 2)))
        discrim_model.add(BatchNormalization())
        discrim_model.add(LeakyReLU(alpha = 0.05))
        discrim_model.add(Dropout(rate = 0.10))
        #Layer 4:
        #Input: 16x16x128
        #Output: 8x8x256
        discrim_model.add(Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = "same"))
        discrim_model.add(MaxPooling2D(pool_size=(2, 2)))
        discrim_model.add(BatchNormalization())
        discrim_model.add(LeakyReLU(alpha = 0.05))
        discrim_model.add(Dropout(rate = 0.10))
        
        #Output Layer: 
        #Input: (8*8*256, )
        #Output: 1-dimensional probability
        discrim_model.add(Flatten())
        discrim_model.add(Dense(1, activation = "sigmoid", name='classif_discrim'))

        print("Architecture for discriminator network from model {}:".format(self.name))
        print(discrim_model.summary())

        ##Feed the discrimator with an image
        img = Input(shape = self.img_shape)
        classify_img = discrim_model(img)

        return Model(inputs = img, outputs = classify_img)

        ## Generator network
    def build_generator_2(self):
        gen_model = Sequential()

        #Layer 1:
        #Input: random noise = 100
        #Output: 8x8x256
        gen_model.add(Dense(units = 8*8*self.nf4, activation = "relu", input_dim = self.latent_dim, name='latent_dim_sample'))
        gen_model.add(Reshape((8,8, self.nf4)))

        #Layer 2
        #Input:  8x8x256
        #Output: 16x16x128
        gen_model.add(UpSampling2D())
        gen_model.add(Conv2D(filters = self.nf3, kernel_size = (3,3), strides=(1, 1), padding = "same"))
        gen_model.add(BatchNormalization())
        gen_model.add(Activation("relu"))

        #Layer 3
        #Input: 16x16x128
        #Output: 32x32x64
        gen_model.add(UpSampling2D())
        gen_model.add(Conv2D(filters = self.nf2, kernel_size = (3,3), strides=(1, 1), padding = "same"))
        gen_model.add(BatchNormalization())
        gen_model.add(Activation("relu"))
        
        #Layer 4
        #Input: 32x32x64
        #Output: 64x64x32
        gen_model.add(UpSampling2D())
        gen_model.add(Conv2D(filters = self.nf1, kernel_size = (3,3), strides=(1, 1), padding = "same"))
        gen_model.add(BatchNormalization())
        gen_model.add(Activation("relu"))

        #Layer 5
        #Input: 64x64x32
        #Output: 128x128x32
        gen_model.add(UpSampling2D())
        gen_model.add(Conv2D(filters = self.nf1, kernel_size = (3,3), strides=(1, 1), padding = "same"))
        gen_model.add(BatchNormalization())
        gen_model.add(Activation("relu"))
        
        #Output Layer:
        #Input: 128x128x32
        #Output: 128x128x3
        gen_model.add(Conv2D(filters = self.channels, kernel_size = 3, padding = "same"))
        gen_model.add(Activation("sigmoid" , name='generated_image'))

        print("Architecture for generator network:")
        print(gen_model.summary())

        latent_noise = Input(shape=(self.latent_dim,))
        generated_img = gen_model(latent_noise)

        return Model(inputs = latent_noise, outputs = generated_img)    
    #####################################################################################

###################################### Model 3 ######################################
        
        ## Discriminator network
    def build_discriminator_3(self):
        discrim_model = Sequential()
        #Layer 1:
        #Input: 128x128x3
        #Output: 64x64x32
        discrim_model.add(Conv2D(filters = self.nf1, kernel_size = (3, 3), strides = (2, 2), input_shape = self.img_shape, padding = "same"))
        discrim_model.add(LeakyReLU(alpha = 0.2))
        discrim_model.add(Dropout(rate = 0.25))
        #Layer 2:
        #Input: 64x64x32
        #Output: 32x32x64
        discrim_model.add(Conv2D(filters = self.nf2, kernel_size = (3, 3), strides = (2, 2), padding = "same"))
        discrim_model.add(BatchNormalization(momentum = 0.9))
        discrim_model.add(LeakyReLU(alpha = 0.2))
        discrim_model.add(Dropout(rate = 0.25))
        #Layer 3:
        #Input: 32x32x64
        #Output: 16x16x128
        discrim_model.add(Conv2D(filters = self.nf3, kernel_size = (3, 3), strides = (2, 2), padding = "same"))
        discrim_model.add(BatchNormalization(momentum = 0.9))
        discrim_model.add(LeakyReLU(alpha = 0.2))
        discrim_model.add(Dropout(rate = 0.25))
        #Layer 4:
        #Input: 16x16x128
        #Output: 8x8x256
        discrim_model.add(Conv2D(filters = 256, kernel_size = 3, strides = 2, padding = "same"))
        discrim_model.add(BatchNormalization(momentum = 0.9))
        discrim_model.add(LeakyReLU(alpha = 0.2))
        discrim_model.add(Dropout(rate = 0.25))
        
        #Output Layer: 
        #Input: (8*8*256, )
        #Output: 1-dimensional probability
        discrim_model.add(Flatten())
        discrim_model.add(Dense(1, activation = "sigmoid", name='classif_discrim'))

        print("Architecture for discriminator network from model {}:".format(self.name))
        print(discrim_model.summary())

        ##Feed the discrimator with an image
        img = Input(shape = self.img_shape)
        classify_img = discrim_model(img)

        return Model(inputs = img, outputs = classify_img)

        ## Generator network
    def build_generator_3(self):
        gen_model = Sequential()

        #Layer 1:
        #Input: random noise = 100
        #Output: 8x8x256
        gen_model.add(Dense(units = 8*8*self.nf4, activation = "relu", input_dim = self.latent_dim, name='latent_dim_sample'))
        gen_model.add(Reshape((8,8, self.nf4)))

        #Layer 2
        #Input:  8x8x256
        #Output: 16x16x128
        gen_model.add(UpSampling2D())
        gen_model.add(Conv2D(filters = self.nf3, kernel_size = (3,3), strides=(1, 1), padding = "same"))
        gen_model.add(BatchNormalization(momentum = 0.9))
        gen_model.add(Activation("relu"))

        #Layer 3
        #Input: 16x16x128
        #Output: 32x32x64
        gen_model.add(UpSampling2D())
        gen_model.add(Conv2D(filters = self.nf2, kernel_size = (3,3), strides=(1, 1), padding = "same"))
        gen_model.add(BatchNormalization(momentum = 0.9))
        gen_model.add(Activation("relu"))
        
        #Layer 4
        #Input: 32x32x64
        #Output: 64x64x32
        gen_model.add(UpSampling2D())
        gen_model.add(Conv2D(filters = self.nf1, kernel_size = (3,3), strides=(1, 1), padding = "same"))
        gen_model.add(BatchNormalization(momentum = 0.9))
        gen_model.add(Activation("relu"))

        #Layer 5
        #Input: 64x64x32
        #Output: 128x128x32
        gen_model.add(UpSampling2D())
        gen_model.add(Conv2D(filters = self.nf1, kernel_size = (3,3), strides=(1, 1), padding = "same"))
        gen_model.add(BatchNormalization(momentum = 0.9))
        gen_model.add(Activation("relu"))
        
        #Output Layer:
        #Input: 128x128x32
        #Output: 128x128x3
        gen_model.add(Conv2D(filters = self.channels, kernel_size = 3, padding = "same"))
        gen_model.add(Activation("sigmoid", name='generated_image'))

        print("Architecture for generator network:")
        print(gen_model.summary())

        latent_noise = Input(shape=(self.latent_dim,))
        generated_img = gen_model(latent_noise)

        return Model(inputs = latent_noise, outputs = generated_img)
    #####################################################################################
    
    def build_dcgan(self):
        ## The generator gets as input noise sampled from the latent space vector Z
        Z = Input(shape = (self.latent_dim, ), name='latent_dim_sample')
        ## generate image with noisy latent input vector Z
        generated_img = self.generator(Z)
        ## The discriminator takes the generated image as input and either classifies them as real or fake
        discrim_classify = self.discriminator(generated_img)
        ## Combine generator and discriminator to simultaneously optimize the weights as a stacked (adversarial) model overall
        stacked_G_D = Model(inputs = Z, outputs = discrim_classify)
        ## In DCGAN only the generator will be trained to create images look-alike the "real" images to fool the discriminator: freeze D-weights
        self.discriminator.trainable = False
        stacked_G_D.compile(loss = "binary_crossentropy", optimizer = self.optimizer)
        print('Printing out stacked model %s' %self.name)
        print(stacked_G_D.summary())
        return stacked_G_D
    
    ## Helper for scaling and unscaling:
    def scale(self, x, out_range = (-1, 1)):
        domain = np.min(x), np.max(x)
        # a)scale data such that its symmetric around 0
        y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
        # b)rescale data such that it falls into desired output range
        y = y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

        return y

    def unscale(self, y, x, out_range = (-1, 1)):
        domain = np.min(x), np.max(x)
        # undo b)
        z = (y - (out_range[1] + out_range[0]) / 2) / (out_range[1] - out_range[0])
        # undo a)
        z = z * (domain[1] - domain[0]) + (domain[1] + domain[0]) / 2

        return z


    def train(self, data, epochs = 100, batch_size = 10, save_intervals = 20,
              init_train=True, start_epoch=0, cycle=1):
        
        print('Training  %s model with following architecture:' %self.name)
        print(self.stacked_G_D.summary())
        
        final_images_stacked = data.astype('float32')
        domain = np.min(final_images_stacked), np.max(final_images_stacked)
        if not domain == (-1,1) and self.name == 'DCGAN_1':
            X_train = self.scale(x = data.astype('float32'), out_range=(-1,1))
        elif not domain == (0,1) and self.name == 'DCGAN_2':
            X_train = self.scale(x = data.astype('float32'), out_range=(0,1))
        else:
            X_train = final_images_stacked
            
        #adversarial truth:
        valid = np.ones(shape = (batch_size, 1))
        fake = np.zeros(shape = (batch_size, 1))
        
        if init_train:
            epoch_iterator = np.arange(start=0, stop=epochs+1)
        else:
            epoch_iterator = np.arange(start=start_epoch+1, stop=start_epoch+epochs+1)
        
        history_list = []
        for epoch in epoch_iterator:

            # ---------------------
            #  Train Variational Autoencoder
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
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Indirectly train the generator through the adversarial (stacked) model:
                # Pass noise to the adversarial model and mislabel everything as if they were images taken from the true database,
                # When they will be generated by the generator.
            g_loss = self.stacked_G_D.train_on_batch(noise, valid)

            # Print the progress
            print ("Epoch: %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            history_list.append("Epoch: %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            # If at save interval => save generated image samples + model weights
            if epoch % save_intervals == 0:
                #create 2x2 images
                self.save_imgs(epoch = epoch, final_images_stacked = final_images_stacked, domain=domain)
                self.generator.save_weights(filepath = "../model/{}/epoch_".format(self.name) + str(epoch) + "_generator.h5")
                self.discriminator.save_weights(filepath = "../model/{}/epoch_".format(self.name) + str(epoch) + "_discriminator.h5")

            #If last epoch save models and generate 10 images in full mode:
            if epoch == cycle*epochs:
                final_noises = np.random.normal(0, 1, (10, self.latent_dim))
                final_gen_images = self.generator.predict(final_noises)
                if not domain == (-1,1) and self.name == 'DCGAN_1':
                    final_gen_images = self.unscale(y = final_gen_images, x = final_images_stacked, out_range=(-1,1))
                    #else is sowieso in output range (0,1) wegen sigmoid
                for i in range(10):
                    plt.imshow(final_gen_images[i, :, :, :], interpolation = "nearest")
                    plt.savefig("../model/%s/images/epoch_%d_final_generated_images_%d.jpg" % (self.name, epoch, i))


    def save_imgs(self, epoch, final_images_stacked, domain):
        r, c = 2, 2
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        
        #rescale to the input range
        if not domain == (-1,1):
            gen_imgs = self.unscale(y = gen_imgs, x = final_images_stacked, out_range=(0,1))
            # else will be anyways in range (0,1) because of sigmoid activation
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis("off")
                cnt += 1
        fig.savefig("../model/%s/image_%d.jpg" % (self.name, epoch))
        plt.close(fig)
