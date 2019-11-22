"""
@title vae.py
@author: Tuan Le
@email: tuanle@hotmail.de

This script builds a Convolutional Variationel Autoencoder using keras framework with tensorflow backend.
"""

from keras.layers import Input, Dense, Reshape, Flatten, Lambda, Dropout
from keras.layers import BatchNormalization, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D #https://github.com/keras-team/keras/issues/7307
from keras.models import Model
from keras.optimizers import Adadelta
from keras import backend as K
import numpy as np
from keras.losses import mse, binary_crossentropy
import os
import matplotlib.pyplot as plt
#https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
#https://distill.pub/2016/deconv-checkerboard/

class VAE():
    def __init__(self, name='VAE_1', use_mse=True):
        assert any(name.upper() in item for item in ['VAE_1', 'VAE_2', 'VAE_3', 'VAE_4']), 'Inserted <name>: "{}" is not provided in the list [VAE_1, VAE_2, VAE_3, VAE_4]'.format(name)
        # Parameters
        self.use_mse = use_mse
        self.name = name.upper()
        self.rows = 128
        self.cols = 128
        self.channels = 3
        self.img_shape = (self.rows, self.cols, self.channels)
        self.optimizer = Adadelta()
        self.intermediate_dim_1 = 256
        self.intermediate_dim_2 = 128
        self.latent_dim = 100
        self.shape_info = None
        self.random_normal_mean = 0.0
        self.random_normal_stddev = 1.0
        ## define number of convolutional filters
        self.z_mean = None
        self.z_log_var = None
        self.z = None
        self.nf1 = 256
        self.nf2 = 128
        self.nf3 = 64
        self.nf4 = 32
        self.nf5 = 16
        self.nf6 = 8
        if not os.path.exists('../model/{}'.format(self.name)):
            os.makedirs('../model/{}'.format(self.name))
            os.makedirs('../model/{}/images'.format(self.name))
        ## Placeholders
        self.vae_input = None
        self.vae_output = None
        #initiate encoder and decoder model
        if self.name == 'VAE_1':
            self.encoder = self.build_conv_encoder_1()
            self.decoder = self.build_deconv_decoder_1()
        elif self.name == 'VAE_2':
            self.encoder = self.build_conv_encoder_2()
            self.decoder = self.build_deconv_decoder_2()
        elif self.name == 'VAE_3':
            self.encoder = self.build_conv_encoder_3()
            self.decoder = self.build_deconv_decoder_3()           
        elif self.name == 'VAE_4':
            self.encoder = self.build_conv_encoder_4()
            self.decoder = self.build_deconv_decoder_4()  
        else:
            print('Model %s cannot be found.' %self.name)
        #init stacked vae-model
        self.vae = self.build_vae()
        #configure custom VAE loss to vae-model
        vae_loss = self.vae_loss(use_mse=use_mse)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer=self.optimizer)


    # reparameterization trick
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # z = z_mean + sqrt(var)*eps
    def sampling(self, args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim),
                                  mean=self.random_normal_mean, stddev=self.random_normal_stddev)
        # Sample z ~ Q(z|X)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    ########################## Architectures regarding the VAE ##########################
    
    ###################################### Model 1 ######################################
    def build_conv_encoder_1(self):
        #Layer 1:
        #Input: 128x128x3
        #Conv:64x64x256
        #Out: 64x64x256
        input_img = Input(shape=self.img_shape, name='image_input')
        conv1 = Conv2D(filters=self.nf1, kernel_size=(3, 3), strides=(2,2), activation='relu', padding='same', name='1_conv_256')(input_img)
        bn1 = BatchNormalization(name='enc_batch_normalization_1')(conv1)
        #save input_img tensor for later usage:
        self.vae_input = input_img

        #Layer 2:
        #Input: 64x64x256
        #Conv: 32x32x128
        #Out: 32x32x128
        conv2 = Conv2D(filters=self.nf2, kernel_size=(3, 3), strides=(2,2), activation='relu', padding='same', name='2_conv_128')(bn1)
        bn2 = BatchNormalization(name='enc_batch_normalization_2')(conv2)

        #Layer 3:
        #Input: 32x32x128
        #Conv: 16x16x64
        #Out: 16x16x64
        conv3 = Conv2D(filters=self.nf3, kernel_size=(3, 3), strides=(2,2), activation='relu', padding='same', name='3_conv_64')(bn2)
        bn3 = BatchNormalization(name='enc_batch_normalization_3')(conv3)

        #Layer 3.1:
        #Input: 16x16x64
        #Conv: 16x16x64
        #Out: 16x16x64
        conv3_1 = Conv2D(filters=self.nf3, kernel_size=(3, 3), strides=(1,1), activation='relu', padding='same', name='3_1_conv_64')(bn3)
        bn3_1 = BatchNormalization(name='enc_batch_normalization_3_1')(conv3_1)

        #Layer 4:
        #Input: 16x16x64
        #Conv: 8x8x32
        #Out: 8x8x32
        conv4 = Conv2D(filters=self.nf4, kernel_size=(3, 3), strides=(2,2), activation='relu', padding='same', name='4_conv_32')(bn3_1)
        bn4 = BatchNormalization(name='enc_batch_normalization_4')(conv4)

        #Layer 4.1:
        #Input: 8x8x32
        #Conv: 8x8x32
        #Out: 8x8x32
        conv4_1 = Conv2D(filters=self.nf4, kernel_size=(3, 3), strides=(1,1), activation='relu', padding='same', name='4_1_conv_32')(bn4)
        bn4_1 = BatchNormalization(name='enc_batch_normalization_4_1')(conv4_1)

        #Layer 5:
        #Input: 8x8x32
        #Conv: 4x4x16
        #Out: 4x4x16
        conv5 = Conv2D(filters=self.nf5, kernel_size=(3, 3), strides=(2,2), activation='relu', padding='same', name='5_conv_16')(bn4_1)
        bn5 = BatchNormalization(name='enc_batch_normalization_5')(conv5)

        #Layer 6:
        #Input: 4x4x16
        #Conv: 4x4x16
        #Out: 4x4x16
        conv6 = Conv2D(filters=self.nf5, kernel_size=(3, 3), strides=(1,1), activation='relu', padding='same', name='6_conv_16')(bn5)
        bn6 = BatchNormalization(name='enc_batch_normalization_6')(conv6)

        ##save shape info from last three-dimensional layer for later upsampling
        self.shape_info = K.int_shape(bn6)
        #here it is (None, 4, 4, 16)

        #Flatten-Layer
        flatten_last_layer = Flatten(name='flatten_3_dim_to_1')(bn6)
        #Build 2 intermediate dense layers
        intermediate_1 = Dense(units=self.intermediate_dim_1, activation='relu', name='intermediate_dim_1')(flatten_last_layer)
        bn7 = BatchNormalization(name='enc_batch_normalization_7')(intermediate_1)
        
        #get mean and variance 'branches'
        ## save object reference because later needed
        self.z_mean = Dense(units=self.latent_dim, name='z_mean')(bn7)
        self.z_log_var = Dense(units=self.latent_dim, name='z_log_var')(bn7)

        #use reparameterization trick to push the sampling out as input
        # Sample z ~ Q(z|X)
        self.z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([self.z_mean, self.z_log_var])

        #instantiate the encoder model
        encoder_model = Model(inputs=input_img, outputs=[self.z_mean, self.z_log_var, self.z], name='encoder_model')
        print('Printing out the encoder from model: %s:' %self.name)
        print(encoder_model.summary())

        return encoder_model

    def build_deconv_decoder_1(self):
        #Input-FC-Layer
        #Input: Fully Connected: 100-dimensional vector [latent_dimension]
        #Hidden Layer Fully Connected: 256-dimensional vector [intermediate_dimension]
        #Hidden Layer Fully Connected: 256-dimensional vector [to_be_reshaped] -- from self.shape_info (None, 4,4,16)
        #Out: 4x4x16
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        rev_bn00 = BatchNormalization(name='dec_batch_normalization_00')(latent_inputs)
        decoder_hidden_1 = Dense(units=self.intermediate_dim_1, activation='relu', name='decoder_intermediate_1')(rev_bn00)
        rev_bn01 = BatchNormalization(name='dec_batch_normalization_01')(decoder_hidden_1)
        decoder_upsample = Dense(units=self.shape_info[1]*self.shape_info[2]*self.shape_info[3], name='reverse_flatten_3_to_1')(rev_bn01)
        decoder_reshape = Reshape(self.shape_info[1:], name='Reshape_get_last_5_conv_encoderlayer')(decoder_upsample)

        ## Layer 1:
        #Input: 4x4x16
        #Output: 4x4x16
        decoder_conv_1 = Conv2D(self.nf5, kernel_size=(3,3), strides=(1, 1), padding='same',
                                 activation='relu',name='Deconv_filter_1_16')(decoder_reshape)
        rev_bn1 = BatchNormalization(name='dec_batch_normalization_1')(decoder_conv_1)

        #apply deconvolution operations with upsamling
        #Layer 2
        #Input: 4x4x16
        #Deconv (out): 8x8x32
        ups1 = UpSampling2D(size=(2,2), name='dec_upsample_1')(rev_bn1)
        decoder_deconv_2 = Conv2D(self.nf4, kernel_size=(3,3), strides=(1, 1), padding='same',
                                 activation='relu',name='Deconv_filter_2_32')(ups1)
        rev_bn2 = BatchNormalization(name='dec_batch_normalization_2')(decoder_deconv_2)

        #Layer 2.1
        #Input: 8x8x32
        #Deconv (out): 8x8x32
        decoder_deconv_2_1 = Conv2D(self.nf4, kernel_size=(3,3), strides=(1, 1), padding='same',
                                 activation='relu',name='Deconv_filter_2_1_32')(rev_bn2)
        rev_bn2_1 = BatchNormalization(name='dec_batch_normalization_2_1')(decoder_deconv_2_1)

        #Layer 3
        #Input: 8x8x32
        #Deconv (out): 16x16x64
        ups2 = UpSampling2D(size=(2,2), name='dec_upsample_2')(rev_bn2_1)
        decoder_deconv_3 = Conv2D(self.nf3, kernel_size=(3,3), strides=(1, 1), padding='same',
                                 activation='relu',name='Deconv_filter_3_64')(ups2)
        rev_bn3 = BatchNormalization(name='dec_batch_normalization_3')(decoder_deconv_3)

        #Layer 3.1
        #Input: 16x16x64
        #Deconv (out): 16x16x64
        decoder_deconv_3_1 = Conv2D(self.nf3, kernel_size=(3,3), strides=(1, 1), padding='same',
                                 activation='relu',name='Deconv_filter_3_1_64')(rev_bn3)
        rev_bn3_1 = BatchNormalization(name='dec_batch_normalization_3_1')(decoder_deconv_3_1)

        #Layer 4
        #Input: 16x16x64
        #Deconv (out): 32x32x128
        ups3 = UpSampling2D(size=(2,2), name='dec_upsample_3')(rev_bn3_1)
        decoder_deconv_4 =Conv2D(self.nf2, kernel_size=(3,3), strides=(1, 1), padding='same',
                                 activation='relu',name='Deconv_filter_4_128')(ups3)
        rev_bn4 = BatchNormalization(name='dec_batch_normalization_4')(decoder_deconv_4)

        #Layer 5
        #Input: 32x32x128
        #Deconv (out): 64x64x256
        ups4 = UpSampling2D(size=(2,2), name='dec_upsample_4')(rev_bn4)
        decoder_deconv_5 = Conv2D(self.nf1, kernel_size=(3,3), strides=(1, 1), padding='same',
                                 activation='relu',name='Deconv_filter_5_256')(ups4)
        rev_bn5 = BatchNormalization(name='dec_batch_normalization_5')(decoder_deconv_5)

        #Layer 6
        #Input: 64x64x256
        #Conv (out): 128x128x64
        ups5 = UpSampling2D(size=(2,2), name='dec_upsample_5')(rev_bn5)
        decoder_deconv_6 = Conv2D(self.nf3, kernel_size=(3,3), strides=(1, 1), padding='same',
                                 activation='relu',name='Deconv_filter_6_64')(ups5)
        rev_bn6 = BatchNormalization(name='dec_batch_normalization_6')(decoder_deconv_6)

        #Layer 7
        #Input: 128x128x64
        #Conv (out): 128x128x32
        decoder_deconv_7 = Conv2D(self.nf4, kernel_size=(3,3), strides=(1, 1), padding='same',
                                 activation='relu',name='Deconv_filter_7_32')(rev_bn6)
        rev_bn7 = BatchNormalization(name='dec_batch_normalization_7')(decoder_deconv_7)

        #Layer 8
        #Input: 128x128x32
        #Conv (out): 128x128x16
        decoder_deconv_8 = Conv2D(self.nf5, kernel_size=(3,3), strides=(1, 1), padding='same',
                                 activation='relu',name='Deconv_filter_8_16')(rev_bn7)
        rev_bn8 = BatchNormalization(name='dec_batch_normalization_8')(decoder_deconv_8)

        #Ouput Decoder Layer
        #Input: 128x128x16
        #Output: 128x128x3 == self.img_shape
        decoded_image = Conv2D(self.channels, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                     activation='sigmoid', name='created_image')(rev_bn8)
        ##out_range is because of sigmoid activation already (0,1)
        #instantiate the decoder model
        decoder_model = Model(latent_inputs, decoded_image, name='decoder_model')
        print('Printing out the decoder from model: %s:' %self.name)
        print(decoder_model.summary())

        return decoder_model
    #####################################################################################
    
    ###################################### Model 2 ######################################
    def build_conv_encoder_2(self):
        ## pooling operations in case needed:
        pool1 = MaxPooling2D(pool_size=(2, 2), name='1_max_pooling_2_2')
        pool2 = MaxPooling2D(pool_size=(2, 2), name='2_max_pooling_2_2')
        pool3 = MaxPooling2D(pool_size=(2, 2), name='3_max_pooling_2_2')
        pool4 = MaxPooling2D(pool_size=(2, 2), name='4_max_pooling_2_2')
        pool5 = MaxPooling2D(pool_size=(2, 2), name='5_max_pooling_2_2')

        #Layer 1:
        #Input: 128x128x3
        #Conv:64x64x256
        #Out: 64x64x256
        input_img = Input(shape=self.img_shape, name='image_input')
        conv1 = Conv2D(filters=self.nf1, kernel_size=(3, 3), strides=(1,1), activation='relu', padding='same', name='1_conv_256')(input_img)
        pool_1 = pool1(conv1)
        bn1 = BatchNormalization(name='enc_batch_normalization_1')(pool_1)
        #save input_img tensor for later usage:
        self.vae_input = input_img

        #Layer 2:
        #Input: 64x64x256
        #Conv: 32x32x128
        #Out: 32x32x128
        conv2 = Conv2D(filters=self.nf2, kernel_size=(3, 3), strides=(1,1), activation='relu', padding='same', name='2_conv_128')(bn1)
        pool_2 = pool2(conv2)
        bn2 = BatchNormalization(name='enc_batch_normalization_2')(pool_2)

        #Layer 3:
        #Input: 32x32x128
        #Conv: 16x16x64
        #Out: 16x16x64
        conv3 = Conv2D(filters=self.nf3, kernel_size=(3, 3), strides=(1,1), activation='relu', padding='same', name='3_conv_64')(bn2)
        pool_3 = pool3(conv3)
        bn3 = BatchNormalization(name='enc_batch_normalization_3')(pool_3)

        #Layer 4:
        #Input: 16x16x64
        #Conv: 8x8x32
        #Out: 8x8x32
        conv4 = Conv2D(filters=self.nf4, kernel_size=(3, 3), strides=(1,1), activation='relu', padding='same', name='4_conv_32')(bn3)
        pool_4 = pool4(conv4)
        bn4 = BatchNormalization(name='enc_batch_normalization_4')(pool_4)

        #Layer 5:
        #Input: 8x8x32
        #Conv: 4x4x16
        #Out: 4x4x16
        conv5 = Conv2D(filters=self.nf5, kernel_size=(3, 3), strides=(1,1), activation='relu', padding='same', name='5_conv_16')(bn4)
        pool_5 = pool5(conv5)
        bn5 = BatchNormalization(name='enc_batch_normalization_5')(pool_5)

        ##save shape info from last three-dimensional layer for later upsampling
        self.shape_info = K.int_shape(bn5)
        #here it is (None, 4, 4, 16)

        #Flatten-Layer
        flatten_last_layer = Flatten(name='flatten_3_dim_to_1')(bn5)
        #Build 2 intermediate dense layers
        intermediate_1 = Dense(units=self.intermediate_dim_1, activation='relu', name='intermediate_dim_1')(flatten_last_layer)
        bn7 = BatchNormalization(name='enc_batch_normalization_7')(intermediate_1)
        #intermediate_2 = Dense(units=self.intermediate_dim_2, activation='relu', name='intermediate_dim_2')(bn7)
        #bn8 = BatchNormalization(name='enc_batch_normalization_8')(intermediate_2)
        #get mean and variance 'branches'
        ## save object reference because later needed
        self.z_mean = Dense(units=self.latent_dim, name='z_mean')(bn7)
        self.z_log_var = Dense(units=self.latent_dim, name='z_log_var')(bn7)

        #use reparameterization trick to push the sampling out as input
        # Sample z ~ Q(z|X)
        self.z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([self.z_mean, self.z_log_var])

        #instantiate the encoder model
        encoder_model = Model(inputs=input_img, outputs=[self.z_mean, self.z_log_var, self.z], name='encoder_model')
        print('Printing out the encoder from model: %s:' %self.name)
        print(encoder_model.summary())

        return encoder_model

    def build_deconv_decoder_2(self):
        #Input-FC-Layer
        #Input: Fully Connected: 100-dimensional vector [latent_dimension]
        #Hidden Layer Fully Connected: 256-dimensional vector [intermediate_dimension]
        #Hidden Layer Fully Connected: 256-dimensional vector [to_be_reshaped] -- from self.shape_info (None, 4,4,16)
        #Out: 4x4x16
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        rev_bn00 = BatchNormalization(name='dec_batch_normalization_00')(latent_inputs)
        decoder_hidden_1 = Dense(units=self.intermediate_dim_1, activation='relu', name='decoder_intermediate_1')(rev_bn00)
        rev_bn01 = BatchNormalization(name='dec_batch_normalization_01')(decoder_hidden_1)
        #decoder_hidden_2 = Dense(units=self.intermediate_dim_2, activation='relu', name='decoder_intermediate_2')(rev_bn01)
        #rev_bn02 = BatchNormalization(name='dec_batch_normalization_02')(decoder_hidden_2)
        decoder_upsample = Dense(units=self.shape_info[1]*self.shape_info[2]*self.shape_info[3], name='reverse_flatten_3_to_1')(rev_bn01)
        decoder_reshape = Reshape(self.shape_info[1:], name='Reshape_get_last_5_conv_encoderlayer')(decoder_upsample)

        ## Layer 1:
        #Input: 4x4x16
        #Output: 4x4x16
        decoder_conv_1 = Conv2D(self.nf5, kernel_size=(3,3), strides=(1, 1), padding='same',
                                 activation='relu',name='Deconv_filter_1_16')(decoder_reshape)
        rev_bn1 = BatchNormalization(name='dec_batch_normalization_1')(decoder_conv_1)

        #apply deconvolution operations with upsamling
        #Layer 2
        #Input: 4x4x16
        #Deconv (out): 8x8x32
        ups1 = UpSampling2D(size=(2,2), name='dec_upsample_1')(rev_bn1)
        decoder_deconv_2 = Conv2D(self.nf4, kernel_size=(3,3), strides=(1, 1), padding='same',
                                 activation='relu',name='Deconv_filter_2_32')(ups1)
        rev_bn2 = BatchNormalization(name='dec_batch_normalization_2')(decoder_deconv_2)

        #Layer 2.1
        #Input: 8x8x32
        #Deconv (out): 8x8x32
        decoder_deconv_2_1 = Conv2D(self.nf4, kernel_size=(3,3), strides=(1, 1), padding='same',
                                 activation='relu',name='Deconv_filter_2_1_32')(rev_bn2)
        rev_bn2_1 = BatchNormalization(name='dec_batch_normalization_2_1')(decoder_deconv_2_1)

        #Layer 3
        #Input: 8x8x32
        #Deconv (out): 16x16x64
        ups2 = UpSampling2D(size=(2,2), name='dec_upsample_2')(rev_bn2_1)
        decoder_deconv_3 = Conv2D(self.nf3, kernel_size=(3,3), strides=(1, 1), padding='same',
                                 activation='relu',name='Deconv_filter_3_64')(ups2)
        rev_bn3 = BatchNormalization(name='dec_batch_normalization_3')(decoder_deconv_3)

        #Layer 3.1
        #Input: 16x16x64
        #Deconv (out): 16x16x64
        decoder_deconv_3_1 = Conv2D(self.nf3, kernel_size=(3,3), strides=(1, 1), padding='same',
                                 activation='relu',name='Deconv_filter_3_1_64')(rev_bn3)
        rev_bn3_1 = BatchNormalization(name='dec_batch_normalization_3_1')(decoder_deconv_3_1)

        #Layer 4
        #Input: 16x16x64
        #Deconv (out): 32x32x128
        ups3 = UpSampling2D(size=(2,2), name='dec_upsample_3')(rev_bn3_1)
        decoder_deconv_4 =Conv2D(self.nf2, kernel_size=(3,3), strides=(1, 1), padding='same',
                                 activation='relu',name='Deconv_filter_4_128')(ups3)
        rev_bn4 = BatchNormalization(name='dec_batch_normalization_4')(decoder_deconv_4)

        #Layer 5
        #Input: 32x32x128
        #Deconv (out): 64x64x256
        ups4 = UpSampling2D(size=(2,2), name='dec_upsample_4')(rev_bn4)
        decoder_deconv_5 = Conv2D(self.nf1, kernel_size=(3,3), strides=(1, 1), padding='same',
                                 activation='relu',name='Deconv_filter_5_256')(ups4)
        rev_bn5 = BatchNormalization(name='dec_batch_normalization_5')(decoder_deconv_5)

        #Layer 6
        #Input: 64x64x256
        #Conv (out): 128x128x64
        ups5 = UpSampling2D(size=(2,2), name='dec_upsample_5')(rev_bn5)
        decoder_deconv_6 = Conv2D(self.nf3, kernel_size=(3,3), strides=(1, 1), padding='same',
                                 activation='relu',name='Deconv_filter_6_64')(ups5)
        rev_bn6 = BatchNormalization(name='dec_batch_normalization_6')(decoder_deconv_6)

        #Layer 7
        #Input: 128x128x64
        #Conv (out): 128x128x32
        decoder_deconv_7 = Conv2D(self.nf4, kernel_size=(3,3), strides=(1, 1), padding='same',
                                 activation='relu',name='Deconv_filter_7_32')(rev_bn6)
        rev_bn7 = BatchNormalization(name='dec_batch_normalization_7')(decoder_deconv_7)

        #Layer 8
        #Input: 128x128x32
        #Conv (out): 128x128x16
        decoder_deconv_8 = Conv2D(self.nf5, kernel_size=(3,3), strides=(1, 1), padding='same',
                                 activation='relu',name='Deconv_filter_8_16')(rev_bn7)
        rev_bn8 = BatchNormalization(name='dec_batch_normalization_8')(decoder_deconv_8)

        #Ouput Decoder Layer
        #Input: 128x128x16
        #Output: 128x128x3 == self.img_shape
        decoded_image = Conv2D(self.channels, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                     activation='sigmoid', name='created_image')(rev_bn8)
        ##out_range is because of sigmoid activation already (0,1)
        #instantiate the decoder model
        decoder_model = Model(latent_inputs, decoded_image, name='decoder_model')
        print('Printing out the decoder from model: %s:' %self.name)
        print(decoder_model.summary())

        return decoder_model
    #####################################################################################
    
    ###################################### Model 3 ######################################
    def build_conv_encoder_3(self):
        ## pooling operations in case needed:
        pool1 = MaxPooling2D(pool_size=(2, 2), name='1_max_pooling_2_2')
        pool2 = MaxPooling2D(pool_size=(2, 2), name='2_max_pooling_2_2')
        pool3 = MaxPooling2D(pool_size=(2, 2), name='3_max_pooling_2_2')
        pool4 = MaxPooling2D(pool_size=(2, 2), name='4_max_pooling_2_2')
        pool5 = MaxPooling2D(pool_size=(2, 2), name='5_max_pooling_2_2')

        #Layer 1:
        #Input: 128x128x3
        #Conv:64x64x256
        #Out: 64x64x256
        input_img = Input(shape=self.img_shape, name='image_input')
        conv1 = Conv2D(filters=self.nf1, kernel_size=(3, 3), strides=(1,1), padding='same', name='1_conv_256')(input_img)
        act1 = LeakyReLU(alpha=0.05, name='enc_1_leaky_relu')(conv1)
        bn1 = BatchNormalization(name='enc_batch_normalization_1')(act1)
        pool_1 = pool1(bn1)
        #dropout1 = Dropout(rate=0.1, name='enc_1_dropout')(pool_1)

        #save input_img tensor for later usage:
        self.vae_input = input_img

        #Layer 2:
        #Input: 64x64x256
        #Conv: 32x32x128
        #Out: 32x32x128
        conv2 = Conv2D(filters=self.nf2, kernel_size=(3, 3), strides=(1,1), padding='same', name='2_conv_128')(pool_1)
        act2 =  LeakyReLU(alpha=0.05, name='enc_2_leaky_relu')(conv2)
        bn2 = BatchNormalization(name='enc_batch_normalization_2')(act2)
        pool_2 = pool2(bn2)
        #dropout2 = Dropout(rate=0.1, name='enc_2_dropout')(pool_2)

        #Layer 3:
        #Input: 32x32x128
        #Conv: 16x16x64
        #Out: 16x16x64
        conv3 = Conv2D(filters=self.nf3, kernel_size=(3, 3), strides=(1,1), padding='same', name='3_conv_64')(pool_2)
        act3 =  LeakyReLU(alpha=0.05, name='enc_3_leaky_relu')(conv3)
        bn3 = BatchNormalization(name='enc_batch_normalization_3')(act3)
        pool_3 = pool3(bn3)
        #dropout3 = Dropout(rate=0.1, name='enc_3_dropout')(pool_3)

        #Layer 4:
        #Input: 16x16x64
        #Conv: 8x8x32
        #Out: 8x8x32
        conv4 = Conv2D(filters=self.nf4, kernel_size=(3, 3), strides=(1,1), padding='same', name='4_conv_32')(pool_3)
        act4 = LeakyReLU(alpha=0.05, name='enc_4_leaky_relu')(conv4)
        bn4 = BatchNormalization(name='enc_batch_normalization_4')(act4)
        pool_4 = pool4(bn4)
        #dropout4 = Dropout(rate=0.1, name='enc_4_dropout')(pool_4)


        #Layer 5:
        #Input: 8x8x32
        #Conv: 4x4x16
        #Out: 4x4x16
        conv5 = Conv2D(filters=self.nf5, kernel_size=(3, 3), strides=(1,1), padding='same', name='5_conv_16')(pool_4)
        act5 = LeakyReLU(alpha=0.05, name='enc_5_leaky_relu')(conv5)
        bn5 = BatchNormalization(name='enc_batch_normalization_5')(act5)
        pool_5 = pool5(bn5)
        #dropout5 = Dropout(rate=0.1, name='5_dropout')(pool_5)

        ##save shape info from last three-dimensional layer for later upsampling
        self.shape_info = K.int_shape(pool_5)
        #here it is (None, 4, 4, 16)

        #Flatten-Layer
        flatten_last_layer = Flatten(name='flatten_3_dim_to_1')(pool_5)
        #Build 2 intermediate dense layers
        intermediate_1 = Dense(units=self.intermediate_dim_1, name='intermediate_dim_1')(flatten_last_layer)
        act_intermd1 = LeakyReLU(alpha=0.05, name='interm1_leaky_relu')(intermediate_1)
        bn7 = BatchNormalization(name='enc_batch_normalization_7')(act_intermd1)

        intermediate_2 = Dense(units=self.intermediate_dim_2, name='intermediate_dim_2')(bn7)
        act_intermd2 = LeakyReLU(alpha=0.05, name='interm2_leaky_relu')(intermediate_2)
        bn8 = BatchNormalization(name='enc_batch_normalization_8')(act_intermd2)

        #get mean and variance 'branches'
        ## save object reference because later needed
        self.z_mean = Dense(units=self.latent_dim, name='z_mean')(bn8)
        self.z_log_var = Dense(units=self.latent_dim, name='z_log_var')(bn8)

        #use reparameterization trick to push the sampling out as input
        # Sample z ~ Q(z|X)
        self.z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([self.z_mean, self.z_log_var])

        #instantiate the encoder model
        encoder_model = Model(inputs=input_img, outputs=[self.z_mean, self.z_log_var, self.z], name='encoder_model')
        print('Printing out the encoder from model: %s:' %self.name)
        print(encoder_model.summary())

        return encoder_model

    def build_deconv_decoder_3(self):
        #Input-FC-Layer
        #Input: Fully Connected: 100-dimensional vector [latent_dimension]
        #Hidden Layer Fully Connected: 256-dimensional vector [intermediate_dimension]
        #Hidden Layer Fully Connected: 256-dimensional vector [to_be_reshaped] -- from self.shape_info (None, 4,4,16)
        #Out: 4x4x16
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        decoder_hidden_1 = Dense(units=self.intermediate_dim_1, name='decoder_intermediate_1')(latent_inputs)
        act1 = LeakyReLU(alpha=0.05, name='h1_leaky_relu')(decoder_hidden_1)
        rev_bn01 = BatchNormalization(name='dec_batch_normalization_01')(act1)

        decoder_hidden_2 = Dense(units=self.intermediate_dim_2, name='decoder_intermediate_2')(rev_bn01)
        act2 = LeakyReLU(alpha=0.05, name='h2_leaky_relu')(decoder_hidden_2)
        rev_bn02 = BatchNormalization(name='dec_batch_normalization_02')(act2)
        decoder_upsample = Dense(units=self.shape_info[1]*self.shape_info[2]*self.shape_info[3], name='reverse_flatten_3_to_1')(rev_bn02)
        decoder_reshape = Reshape(self.shape_info[1:], name='Reshape_get_last_5_conv_encoderlayer')(decoder_upsample)

        ## LINK:
        #https://stackoverflow.com/questions/48226783/what-is-the-the-difference-between-performing-upsampling-together-with-strided-t

        ## Layer 1:
        #Input: 4x4x16
        #Output: 4x4x16
        decoder_conv_1 = Conv2D(self.nf5, kernel_size=(3,3), strides=(1, 1), padding='same', name='Deconv_filter_1_16')(decoder_reshape)
        act_dec1 = LeakyReLU(alpha=0.05, name='dec_1_leaky_relu')(decoder_conv_1)
        rev_bn1 = BatchNormalization(name='dec_batch_normalization_1')(act_dec1)
        #drop1 = Dropout(rate=0.1)(rev_bn1)

        #apply deconvolution operations with upsamling
        #Layer 2
        #Input: 4x4x16
        #Deconv (out): 8x8x32
        ups1 = UpSampling2D(size=(2,2), name='dec_upsample_1')(rev_bn1)
        decoder_deconv_2 = Conv2D(self.nf4, kernel_size=(3,3), strides=(1, 1), padding='same', name='Deconv_filter_2_32')(ups1)
        act_dec2 = LeakyReLU(alpha=0.05, name='dec_2_leaky_relu')(decoder_deconv_2)
        rev_bn2 = BatchNormalization(name='dec_batch_normalization_2')(act_dec2)
        #drop2 = Dropout(rate=0.1)(rev_bn2)

        #Layer 2.1
        #Input: 8x8x32
        #Deconv (out): 8x8x32
        decoder_deconv_2_1 = Conv2D(self.nf4, kernel_size=(3,3), strides=(1, 1), padding='same', name='Deconv_filter_2_1_32')(rev_bn2)
        act_dec2_1 = LeakyReLU(alpha=0.05, name='dec_2_1_leaky_relu')(decoder_deconv_2_1)
        rev_bn2_1 = BatchNormalization(name='dec_batch_normalization_2_1')(act_dec2_1)
        #drop2_1 = Dropout(rate=0.1)(rev_bn2_1)

        #Layer 3
        #Input: 8x8x32
        #Deconv (out): 16x16x64
        ups2 = UpSampling2D(size=(2,2), name='dec_upsample_2')(rev_bn2_1)
        decoder_deconv_3 = Conv2D(self.nf3, kernel_size=(3,3), strides=(1, 1), padding='same', name='Deconv_filter_3_64')(ups2)
        act_dec3 = LeakyReLU(alpha=0.05, name='dec_3_leaky_relu')(decoder_deconv_3)
        rev_bn3 = BatchNormalization(name='dec_batch_normalization_3')(act_dec3)
        #drop3 = Dropout(rate=0.1)(rev_bn3)

        #Layer 3.1
        #Input: 16x16x64
        #Deconv (out): 16x16x64
        decoder_deconv_3_1 = Conv2D(self.nf3, kernel_size=(3,3), strides=(1, 1), padding='same', name='Deconv_filter_3_1_64')(rev_bn3)
        act_dec3_1 = LeakyReLU(alpha=0.05, name='dec_3_1_leaky_relu')(decoder_deconv_3_1)
        rev_bn3_1 = BatchNormalization(name='dec_batch_normalization_3_1')(act_dec3_1)
        drop3_1 = Dropout(rate=0.1)(rev_bn3_1)

        #Layer 4
        #Input: 16x16x64
        #Deconv (out): 32x32x128
        ups3 = UpSampling2D(size=(2,2), name='dec_upsample_3')(drop3_1)
        decoder_deconv_4 =Conv2D(self.nf2, kernel_size=(3,3), strides=(1, 1), padding='same', name='Deconv_filter_4_128')(ups3)
        act_dec4 = LeakyReLU(alpha=0.05, name='dec_4_leaky_relu')(decoder_deconv_4)
        rev_bn4 = BatchNormalization(name='dec_batch_normalization_4')(act_dec4)
        #drop4 = Dropout(rate=0.1)(rev_bn4)

        #Layer 5
        #Input: 32x32x128
        #Deconv (out): 64x64x256
        ups4 = UpSampling2D(size=(2,2), name='dec_upsample_4')(rev_bn4)
        decoder_deconv_5 = Conv2D(self.nf1, kernel_size=(3,3), strides=(1, 1), padding='same', name='Deconv_filter_5_256')(ups4)
        act_dec5 = LeakyReLU(alpha=0.05, name='dec_5_leaky_relu')(decoder_deconv_5)
        rev_bn5 = BatchNormalization(name='dec_batch_normalization_5')(act_dec5)
        #drop5 = Dropout(rate=0.1)(rev_bn5)

        #Layer 6
        #Input: 64x64x256
        #Conv (out): 128x128x64
        ups5 = UpSampling2D(size=(2,2), name='dec_upsample_5')(rev_bn5)
        decoder_deconv_6 = Conv2D(self.nf3, kernel_size=(3,3), strides=(1, 1), padding='same', name='Deconv_filter_6_64')(ups5)
        act_dec6 = LeakyReLU(alpha=0.05, name='dec_6_leaky_relu')(decoder_deconv_6)
        rev_bn6 = BatchNormalization(name='dec_batch_normalization_6')(act_dec6)
        #drop6 = Dropout(rate=0.1)(rev_bn6)

        #Layer 7
        #Input: 128x128x64
        #Conv (out): 128x128x32
        decoder_deconv_7 = Conv2D(self.nf4, kernel_size=(3,3), strides=(1, 1), padding='same', name='Deconv_filter_7_32')(rev_bn6)
        act_dec7 = LeakyReLU(alpha=0.05, name='dec_7_leaky_relu')(decoder_deconv_7)
        rev_bn7 = BatchNormalization(name='dec_batch_normalization_7')(act_dec7)
        #drop7 = Dropout(rate=0.1)(rev_bn7)

        #Layer 8
        #Input: 128x128x32
        #Conv (out): 128x128x16
        decoder_deconv_8 = Conv2D(self.nf5, kernel_size=(3,3), strides=(1, 1), padding='same', name='Deconv_filter_8_16')(rev_bn7)
        act_dec8 = LeakyReLU(alpha=0.05, name='dec_8_leaky_relu')(decoder_deconv_8)
        rev_bn8 = BatchNormalization(name='dec_batch_normalization_8')(act_dec8)
        #drop8 = Dropout(rate=0.1)(rev_bn8)

        #Layer 9
        #Input: 128x128x16
        #Conv (out): 128x128x16
        decoder_deconv_9 = Conv2D(self.nf5, kernel_size=(3,3), strides=(1, 1), padding='same', name='Deconv_filter_9_16')(rev_bn8)
        act_dec9 = LeakyReLU(alpha=0.05, name='dec_9_leaky_relu')(decoder_deconv_9)
        rev_bn9 = BatchNormalization(name='dec_batch_normalization_9')(act_dec9)
        #drop9 = Dropout(rate=0.1)(rev_bn9)

        #Layer 10
        #Input: 128x128x16
        #Conv (out): 128x128x8
        decoder_deconv_10 = Conv2D(self.nf6, kernel_size=(3,3), strides=(1, 1), padding='same', name='Deconv_filter_10_8')(rev_bn9)
        act_dec10 = LeakyReLU(alpha=0.05, name='dec_10_leaky_relu')(decoder_deconv_10)
        rev_bn10 = BatchNormalization(name='dec_batch_normalization_10')(act_dec10)
        #drop10 = Dropout(rate=0.1)(rev_bn10)

        #Ouput Decoder Layer
        #Input: 128x128x8
        #Output: 128x128x3 == self.img_shape
        decoded_image = Conv2D(self.channels, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                     activation='sigmoid', name='created_image')(rev_bn10)

        ##out_range is because of sigmoid activation already (0,1)
        #instantiate the decoder model
        decoder_model = Model(latent_inputs, decoded_image, name='decoder_model')
        print('Printing out the decoder from model: %s:' %self.name)
        print(decoder_model.summary())

        return decoder_model
    #####################################################################################

    ###################################### Model 4 ######################################
    def build_conv_encoder_4(self):
        ## pooling operations in case needed:
        pool1 = MaxPooling2D(pool_size=(2, 2), name='1_max_pooling_2_2')
        pool2 = MaxPooling2D(pool_size=(2, 2), name='2_max_pooling_2_2')
        pool3 = MaxPooling2D(pool_size=(2, 2), name='3_max_pooling_2_2')
        pool4 = MaxPooling2D(pool_size=(2, 2), name='4_max_pooling_2_2')
        pool5 = MaxPooling2D(pool_size=(2, 2), name='5_max_pooling_2_2')

        #Layer 1:
        #Input: 128x128x3
        #Conv:64x64x256
        #Out: 64x64x256
        input_img = Input(shape=self.img_shape, name='image_input')
        conv1 = Conv2D(filters=self.nf1, kernel_size=(3, 3), strides=(1,1), padding='same', name='1_conv_256', activation='relu')(input_img)
        bn1 = BatchNormalization(name='enc_batch_normalization_1')(conv1)
        pool_1 = pool1(bn1)
        dropout1 = Dropout(rate=0.2, name='enc_1_dropout')(pool_1)

        #save input_img tensor for later usage:
        self.vae_input = input_img

        #Layer 2:
        #Input: 64x64x256
        #Conv: 32x32x128
        #Out: 32x32x128
        conv2 = Conv2D(filters=self.nf2, kernel_size=(3, 3), strides=(1,1), padding='same', name='2_conv_128', activation='relu')(dropout1)
        bn2 = BatchNormalization(name='enc_batch_normalization_2')(conv2)
        pool_2 = pool2(bn2)
        dropout2 = Dropout(rate=0.2, name='enc_2_dropout')(pool_2)

        #Layer 3:
        #Input: 32x32x128
        #Conv: 16x16x64
        #Out: 16x16x64
        conv3 = Conv2D(filters=self.nf3, kernel_size=(3, 3), strides=(1,1), padding='same', name='3_conv_64', activation='relu')(dropout2)
        bn3 = BatchNormalization(name='enc_batch_normalization_3')(conv3)
        pool_3 = pool3(bn3)
        dropout3 = Dropout(rate=0.2, name='enc_3_dropout')(pool_3)

        #Layer 4:
        #Input: 16x16x64
        #Conv: 8x8x32
        #Out: 8x8x32
        conv4 = Conv2D(filters=self.nf4, kernel_size=(3, 3), strides=(1,1), padding='same', name='4_conv_32', activation='relu')(dropout3)
        bn4 = BatchNormalization(name='enc_batch_normalization_4')(conv4)
        pool_4 = pool4(bn4)
        dropout4 = Dropout(rate=0.2, name='enc_4_dropout')(pool_4)


        #Layer 5:
        #Input: 8x8x32
        #Conv: 4x4x16
        #Out: 4x4x16
        conv5 = Conv2D(filters=self.nf5, kernel_size=(3, 3), strides=(1,1), padding='same', name='5_conv_16', activation='relu')(dropout4)
        bn5 = BatchNormalization(name='enc_batch_normalization_5')(conv5)
        pool_5 = pool5(bn5)
        dropout5 = Dropout(rate=0.1, name='5_dropout')(pool_5)

        ##save shape info from last three-dimensional layer for later upsampling
        self.shape_info = K.int_shape(dropout5)
        #here it is (None, 4, 4, 16)

        #Flatten-Layer
        flatten_last_layer = Flatten(name='flatten_3_dim_to_1')(dropout5)
        #Build 2 intermediate dense layers
        intermediate_1 = Dense(units=self.intermediate_dim_1, name='intermediate_dim_1', activation='relu')(flatten_last_layer)
        bn7 = BatchNormalization(name='enc_batch_normalization_7')(intermediate_1)
        dropout6 = Dropout(rate=0.1, name='6_dropout')(bn7)
        
        intermediate_2 = Dense(units=self.intermediate_dim_2, name='intermediate_dim_2', activation='relu')(dropout6)
        bn8 = BatchNormalization(name='enc_batch_normalization_8')(intermediate_2)
        
        #get mean and variance 'branches'
        ## save object reference because later needed
        self.z_mean = Dense(units=self.latent_dim, name='z_mean')(bn8)
        self.z_log_var = Dense(units=self.latent_dim, name='z_log_var')(bn8)

        #use reparameterization trick to push the sampling out as input
        # Sample z ~ Q(z|X)
        self.z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([self.z_mean, self.z_log_var])

        #instantiate the encoder model
        encoder_model = Model(inputs=input_img, outputs=[self.z_mean, self.z_log_var, self.z], name='encoder_model')
        print('Printing out the encoder from model: %s:' %self.name)
        print(encoder_model.summary())

        return encoder_model

    def build_deconv_decoder_4(self):
        #Input-FC-Layer
        #Input: Fully Connected: 100-dimensional vector [latent_dimension]
        #Hidden Layer Fully Connected: 256-dimensional vector [intermediate_dimension]
        #Hidden Layer Fully Connected: 256-dimensional vector [to_be_reshaped] -- from self.shape_info (None, 4,4,16)
        #Out: 4x4x16
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        decoder_hidden_1 = Dense(units=self.intermediate_dim_1, name='decoder_intermediate_1', activation='relu')(latent_inputs)
        rev_bn01 = BatchNormalization(name='dec_batch_normalization_01')(decoder_hidden_1)

        decoder_hidden_2 = Dense(units=self.intermediate_dim_2, name='decoder_intermediate_2', activation='relu')(rev_bn01)
        rev_bn02 = BatchNormalization(name='dec_batch_normalization_02')(decoder_hidden_2)
        decoder_upsample = Dense(units=self.shape_info[1]*self.shape_info[2]*self.shape_info[3], name='reverse_flatten_3_to_1')(rev_bn02)
        decoder_reshape = Reshape(self.shape_info[1:], name='Reshape_get_last_5_conv_encoderlayer')(decoder_upsample)

        ## LINK:
        #https://stackoverflow.com/questions/48226783/what-is-the-the-difference-between-performing-upsampling-together-with-strided-t

        ## Layer 1:
        #Input: 4x4x16
        #Output: 4x4x16
        decoder_conv_1 = Conv2D(self.nf5, kernel_size=(3,3), strides=(1, 1), padding='same', name='Deconv_filter_1_16', activation='relu')(decoder_reshape)
        rev_bn1 = BatchNormalization(name='dec_batch_normalization_1')(decoder_conv_1)
        drop1 = Dropout(rate=0.15)(rev_bn1)

        #apply deconvolution operations with upsamling
        #Layer 2
        #Input: 4x4x16
        #Deconv (out): 8x8x32
        ups1 = UpSampling2D(size=(2,2), name='dec_upsample_1')(drop1)
        decoder_deconv_2 = Conv2D(self.nf4, kernel_size=(3,3), strides=(1, 1), padding='same', name='Deconv_filter_2_32', activation='relu')(ups1)
        rev_bn2 = BatchNormalization(name='dec_batch_normalization_2')(decoder_deconv_2)
        drop2 = Dropout(rate=0.15)(rev_bn2)

        #Layer 2.1
        #Input: 8x8x32
        #Deconv (out): 8x8x32
        decoder_deconv_2_1 = Conv2D(self.nf4, kernel_size=(3,3), strides=(1, 1), padding='same', name='Deconv_filter_2_1_32', activation='relu')(drop2)
        rev_bn2_1 = BatchNormalization(name='dec_batch_normalization_2_1')(decoder_deconv_2_1)
        drop2_1 = Dropout(rate=0.15)(rev_bn2_1)

        #Layer 3
        #Input: 8x8x32
        #Deconv (out): 16x16x64
        ups2 = UpSampling2D(size=(2,2), name='dec_upsample_2')(drop2_1)
        decoder_deconv_3 = Conv2D(self.nf3, kernel_size=(3,3), strides=(1, 1), padding='same', name='Deconv_filter_3_64', activation='relu')(ups2)
        rev_bn3 = BatchNormalization(name='dec_batch_normalization_3')(decoder_deconv_3)
        drop3 = Dropout(rate=0.15)(rev_bn3)

        #Layer 3.1
        #Input: 16x16x64
        #Deconv (out): 16x16x64
        decoder_deconv_3_1 = Conv2D(self.nf3, kernel_size=(3,3), strides=(1, 1), padding='same', name='Deconv_filter_3_1_64', activation='relu')(drop3)
        rev_bn3_1 = BatchNormalization(name='dec_batch_normalization_3_1')(decoder_deconv_3_1)
        drop3_1 = Dropout(rate=0.15)(rev_bn3_1)

        #Layer 4
        #Input: 16x16x64
        #Deconv (out): 32x32x128
        ups3 = UpSampling2D(size=(2,2), name='dec_upsample_3')(drop3_1)
        decoder_deconv_4 =Conv2D(self.nf2, kernel_size=(3,3), strides=(1, 1), padding='same', name='Deconv_filter_4_128', activation='relu')(ups3)
        rev_bn4 = BatchNormalization(name='dec_batch_normalization_4')(decoder_deconv_4)
        drop4 = Dropout(rate=0.15)(rev_bn4)

        #Layer 5
        #Input: 32x32x128
        #Deconv (out): 64x64x256
        ups4 = UpSampling2D(size=(2,2), name='dec_upsample_4')(drop4)
        decoder_deconv_5 = Conv2D(self.nf1, kernel_size=(3,3), strides=(1, 1), padding='same', name='Deconv_filter_5_256', activation='relu')(ups4)
        rev_bn5 = BatchNormalization(name='dec_batch_normalization_5')(decoder_deconv_5)
        drop5 = Dropout(rate=0.15)(rev_bn5)

        #Layer 6
        #Input: 64x64x256
        #Conv (out): 128x128x64
        ups5 = UpSampling2D(size=(2,2), name='dec_upsample_5')(drop5)
        decoder_deconv_6 = Conv2D(self.nf3, kernel_size=(3,3), strides=(1, 1), padding='same', name='Deconv_filter_6_64', activation='relu')(ups5)
        rev_bn6 = BatchNormalization(name='dec_batch_normalization_6')(decoder_deconv_6)
        drop6 = Dropout(rate=0.15)(rev_bn6)

        #Layer 7
        #Input: 128x128x64
        #Conv (out): 128x128x32
        decoder_deconv_7 = Conv2D(self.nf4, kernel_size=(3,3), strides=(1, 1), padding='same', name='Deconv_filter_7_32', activation='relu')(drop6)
        rev_bn7 = BatchNormalization(name='dec_batch_normalization_7')(decoder_deconv_7)
        drop7 = Dropout(rate=0.15)(rev_bn7)

        #Layer 8
        #Input: 128x128x32
        #Conv (out): 128x128x16
        decoder_deconv_8 = Conv2D(self.nf5, kernel_size=(3,3), strides=(1, 1), padding='same', name='Deconv_filter_8_16', activation='relu')(drop7)
        rev_bn8 = BatchNormalization(name='dec_batch_normalization_8')(decoder_deconv_8)
        drop8 = Dropout(rate=0.15)(rev_bn8)

        #Layer 9
        #Input: 128x128x16
        #Conv (out): 128x128x16
        decoder_deconv_9 = Conv2D(self.nf5, kernel_size=(3,3), strides=(1, 1), padding='same', name='Deconv_filter_9_16', activation='relu')(drop8)
        rev_bn9 = BatchNormalization(name='dec_batch_normalization_9')(decoder_deconv_9)
        drop9 = Dropout(rate=0.15)(rev_bn9)

        #Layer 10
        #Input: 128x128x16
        #Conv (out): 128x128x8
        decoder_deconv_10 = Conv2D(self.nf6, kernel_size=(3,3), strides=(1, 1), padding='same', name='Deconv_filter_10_8', activation='relu')(drop9)
        rev_bn10 = BatchNormalization(name='dec_batch_normalization_10')(decoder_deconv_10)
        drop10 = Dropout(rate=0.1)(rev_bn10)

        #Ouput Decoder Layer
        #Input: 128x128x8
        #Output: 128x128x3 == self.img_shape
        decoded_image = Conv2D(self.channels, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                     activation='sigmoid', name='created_image')(drop10)

        ##out_range is because of sigmoid activation already (0,1)
        #instantiate the decoder model
        decoder_model = Model(latent_inputs, decoded_image, name='decoder_model')
        print('Printing out the decoder from model: %s:' %self.name)
        print(decoder_model.summary())

        return decoder_model
    #####################################################################################
    #####################################################################################
    
    def build_vae(self):
        #Get the sampled z-vector from latent space. z is component_wise element [-1,1] bc of unit gaussian
        sampled_z = self.encoder(self.vae_input)[2]
        #feed into decoder model to create the image
        self.vae_output = self.decoder(sampled_z)
        #Created stacked variational conv autoencoder : vae = conv_encoder + deconv_decoder
        vae_model = Model(inputs=self.vae_input, outputs=self.vae_output)
        print('Printing out the convolutional variational autoencoder %s model:' %self.name)
        print(vae_model.summary())

        return vae_model

    def vae_loss(self, use_mse):
        if use_mse:
            reconstruction_loss = mse(K.flatten(self.vae_input),
                                      K.flatten(self.vae_output))
        else:
            reconstruction_loss = binary_crossentropy(K.flatten(self.vae_input),
                                                      K.flatten(self.vae_output))

        reconstruction_loss *= self.rows * self.cols
        ## kullback-leibler divergence in closed form
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        #kl_loss = K.mean(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss +  kl_loss)

        return vae_loss

    ## Helper for scaling and unscaling:
    def scale(self, x, out_range = (0, 1)):
        domain = np.min(x), np.max(x)
        # a)scale data such that its symmetric around 0
        y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
        # b)rescale data such that it falls into desired output range
        y = y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

        return y

    def unscale(self, y, x, out_range = (0, 1)):
        domain = np.min(x), np.max(x)
        # undo b)
        z = (y - (out_range[1] + out_range[0]) / 2) / (out_range[1] - out_range[0])
        # undo a)
        z = z * (domain[1] - domain[0]) + (domain[1] + domain[0]) / 2

        return z


    def train(self, data, epochs = 100, batch_size = 32, save_intervals = 50,
              init_train=True, start_epoch=0, cycle=1):
        
        print('Training  %s model with following architecture:' %self.name)
        print(self.vae.summary())
        
        final_images_stacked = data.astype('float32')
        print('Training size: %d' %len(final_images_stacked))
        domain = np.min(final_images_stacked), np.max(final_images_stacked)
        if not domain == (0,1):
            X_train = self.scale(x = data.astype('float32'), out_range=(0,1))
        else:
            X_train = final_images_stacked

        if init_train:
            epoch_iterator = np.arange(start=0, stop=epochs+1)
        else:
            epoch_iterator = np.arange(start=start_epoch+1, stop=start_epoch+epochs+1)

        history_list = []
        for epoch in epoch_iterator:

            # ---------------------
            #  Train Variational Autoencoder 
            # ---------------------

            # Select batch images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # ---------------------

            vae_history = self.vae.train_on_batch(x=imgs, y=None)
            # Print the progress
            print ("Epoch: %d %s loss: %f" % (epoch, self.name, vae_history))
            # save the progress in history object
            history_list.append("Epoch: %d %s loss: %f" % (epoch, self.name, vae_history))
            #If last epoch save models and generate 10 images in full mode:
            if epoch == cycle*epochs:
                final_noises = np.random.normal(0, 1, (10, self.latent_dim))
                final_gen_images = self.decoder.predict(final_noises)
                if not domain == (0,1):
                    final_gen_images = self.unscale(y = final_gen_images, x = final_images_stacked, out_range=(0,1))
                for i in range(10):
                    plt.imshow(final_gen_images[i, :, :, :], interpolation = "nearest")
                    plt.savefig("../model/%s/images/epoch_%d_final_generated_images_%d.jpg" % (self.name, epoch, i))

            if epoch % save_intervals == 0:
                #create 2x2 images
                self.save_imgs(epoch = epoch, final_images_stacked=final_images_stacked, domain=domain)
                #save last weights
                self.encoder.save_weights(filepath = "../model/{}/epoch_".format(self.name) + str(epoch) + "_encoder.h5")
                self.decoder.save_weights(filepath = "../model/{}/epoch_".format(self.name) + str(epoch) + "_decoder.h5")
                self.vae.save_weights(filepath = "../model/{}/epoch_".format(self.name) + str(epoch) + "_vae.h5")

        if not os.path.exists('../model/{}/history.txt'.format(self.name)):
            with open('../model/{}/history.txt'.format(self.name), 'w+') as f:
                for item in history_list:
                    f.write("%s\n" % item)
        else:
            with open('../model/{}/history.txt'.format(self.name), 'a+') as f:
                for item in history_list:
                    f.write("%s\n" % item)

    def save_imgs(self, epoch, final_images_stacked, domain):
        r, c = 2, 2
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.decoder.predict(noise)

        # Rescale images 0 - 1
        if not domain == (0,1):
            gen_imgs = self.unscale(y = gen_imgs, x = final_images_stacked, out_range=(0,1))

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis("off")
                cnt += 1
        fig.savefig("../model/%s/image_%d.jpg" % (self.name, epoch))
        plt.close(fig)
