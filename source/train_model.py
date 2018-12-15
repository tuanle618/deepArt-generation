# -*- coding: utf-8 -*-
"""
@title train_model.py
@author: Tuan Le
@email: tuanle@hotmail.de
"""

from dcgan import DCGAN
from vae import VAE

from data_preprocess import preprocess
import numpy as np
import matplotlib.pyplot as plt
import sys
import gc

def main(model, init_train, start_epoch, cycle, epochs, batch_size, save_intervals):

    model = model.upper()

    if model == 'DCGAN_1':
        my_model = DCGAN(name='DCGAN_1')
    elif model == 'DCGAN_2':
        my_model = DCGAN(name='DCGAN_2')
    elif model == 'DCGAN_3':
        my_model = DCGAN(name='DCGAN_3')
    elif model == 'VAE_1':
        my_model = VAE(name='VAE_1')
    elif model.upper() == 'VAE_2':
        my_model = VAE(name='VAE_2')
    elif model == 'VAE_3':
        my_model = VAE(name='VAE_3')
    elif model == 'VAE_4':
        my_model = VAE(name='VAE_4')
    else:
        print('The selected model {} is not in the list [DCGAN_1, DCGAN_2, DCGAN_3, VAE_1, VAE_2, VAE_3, VAE_4]'.format(model))

    print("Python main programm for generating images using {}".format(model))

    ## preprocess data images if init_train and save the images as pickle file. if not init_train load the saved file
    if init_train:
        print("Start initial process of building the {} model.".format(model))
        print("Do Preprocessing by loading scraped images...")
        ### manually merged into merged_japanese, so take that subdirectory as datapath source:
        if False:
            ## select genre = "yakusha-e"
            image_resized_1 = preprocess(genre_or_style= "yakusha-e", min_vals = [128,128])
            ## select style = "Japanese Art"
            image_resized_2 = preprocess(genre_or_style= "Japanese Art", min_vals = [128,128])
            final_images_stacked = np.vstack((image_resized_1, image_resized_2))
            del image_resized_1, image_resized_2
            gc.collect()
        else:
            final_images_stacked = preprocess(genre_or_style="merged_japanese", min_vals=[128,128])

        ## save the train data such that in the next intermediate steps the preprocess() fnc is not needed, rather load file
        try:
            print("Save preprocessed image data on ../data/train_data.npz in order to retrieve in upcoming training cycles.")
            np.savez_compressed(file="../data/train_data.npz", a=final_images_stacked)
        except:
            print("Could not save train data on machine for upcoming training cycles.")

    else:
        try:
            print("Load preprocessed image data from earlier training cycles.")
            final_images_stacked = np.load(file="../data/train_data.npz")["a"]
        except:
            ### manually merged into merged_japanese, so take that subdirectory as datapath source:
            if False:
                ## select genre = "yakusha-e"
                image_resized_1 = preprocess(genre_or_style= "yakusha-e", min_vals = [128,128])
                ## select style = "Japanese Art"
                image_resized_2 = preprocess(genre_or_style= "Japanese Art", min_vals = [128,128])
                final_images_stacked = np.vstack((image_resized_1, image_resized_2))
                del image_resized_1, image_resized_2
                gc.collect()
            else:
                final_images_stacked = preprocess(genre_or_style="merged_japanese", min_vals=[128,128])

    if init_train:
        print("Start initial training of the {} model:".format(model))
        print("There are {} images provided for training".format(len(final_images_stacked)))
        my_model.train(data = final_images_stacked, epochs = epochs, batch_size = batch_size, save_intervals = save_intervals,
                       init_train=init_train, start_epoch=start_epoch, cycle=cycle)
    else:
        if model in ['DCGAN_1', 'DCGAN_2', 'DCGAN_3']:
            print("Using last epoch {} of generator and discriminator for the stacked {}  model:".format(start_epoch, model))
            generator_weights = "../model/{}/epoch_{}_generator.h5".format(model, start_epoch)
            discrimininator_weights = "../model/{}/epoch_{}_discriminator.h5".format(model, start_epoch)
            #load generator weights
            my_model.generator.load_weights(filepath = generator_weights)
            #load discriminator weights
            my_model.discriminator.load_weights(filepath = discrimininator_weights)
            #train the dcgan with last epoch weights
            print("Training the {} model based on last epoch weights {}.".format(model, start_epoch))
        elif model in ['VAE_1', 'VAE_2', 'VAE_3', 'VAE_4']:
            print("Using last epoch {} of encoder and decoder for the stacked {} model:".format(start_epoch, model))
            encoder_weights = "../model/{}/epoch_{}_encoder.h5".format(model, start_epoch)
            decoder_weights = "../model/{}/epoch_{}_decoder.h5".format(model, start_epoch)
            vae_weights = "../model/{}/epoch_{}_vae.h5".format(model, start_epoch)
            #load encoder weights
            my_model.encoder.load_weights(filepath = encoder_weights)
            #load decoder weights
            my_model.decoder.load_weights(filepath = decoder_weights)
            #load VAE weights
            my_model.vae.load_weights(filepath = vae_weights)
            #train the VAE with last epoch weights
            print("Training the {} model based on last epoch weights {}.".format(model, start_epoch))
        else:
            print('Selected model {} is not available')

        my_model.train(data = final_images_stacked, epochs = epochs, batch_size = batch_size, save_intervals = save_intervals,
                           init_train=init_train, start_epoch=start_epoch, cycle=cycle)

if __name__ == "__main__":
    """
    This script runs the main training programm for the model. Note Model either has to be 'DCGAN', 'VAE', 'VAE_2' or 'VAE_3'
    """
    ### If user inserts via shell console
    if len(sys.argv) >= 2:
        try:
            model = sys.argv[1]
            bool_flag = sys.argv[2].lower() == 'true'
            init_train = bool_flag
            start_epoch = int(sys.argv[3])
            cycle = int(sys.argv[4])
            epochs = int(sys.argv[5])
            batch_size = int(sys.argv[6])
            save_intervals = int(sys.argv[7])
            print("Trying to print out sys.argv")
            print(str(sys.argv))
        except:
            model = "VAE_3"
            init_train = True
            start_epoch = 0
            cycle = 1
            epochs = 100
            batch_size= 32
            save_intervals = 50
    else:
        model = "VAE_2"
        init_train = True
        start_epoch = 0
        cycle = 1
        epochs = 500
        batch_size= 16
        save_intervals = 250

    main(model=model, init_train=init_train, start_epoch=start_epoch,
         cycle=cycle, epochs=epochs, batch_size=batch_size, save_intervals=save_intervals)
