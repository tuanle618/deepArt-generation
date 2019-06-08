#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title data_preprocess.py
@author: Tuan Le
@email: tuanle@hotmail.de
This script loads images in the folder ../data/filepath and preprocess them into a corresponding numpy ndarray
"""
import os
import numpy as np
from skimage import io
from skimage.transform import resize
import os
import gc
from itertools import compress
import argparse

def resize_helper(img, min_vals = [128,128]):
    """
    resize colour images to be square and have certain range
    """
    return resize(image = img, output_shape = (min_vals[0], min_vals[1]), mode="reflect")


def expander(x):
    """
    Add "observation dimension in order to vertical stack images"
    """
    return np.expand_dims(x, axis=0)
    
def preprocess(filepath, min_vals = [128,128], channels_first=True):
    """
    loads images from a path, resizes them to a certain range and finally stacks them
    """
    
    path = "../data/{}/".format(filepath)
    #list all images in ../data/'filepath'/img1.jpg
    
    all_images = [x for x in os.listdir(path) if x.endswith(".jpg") | x.endswith(".png") | x.endswith(".jpeg")]
    all_images = [os.path.join(path, image) for image in all_images]
    all_images_ndarray = list(map(io.imread, all_images))
    all_images_ndarray = list(map(np.array, all_images_ndarray))
    
    show_image = False
    
    if show_image:
        ## show first 5 images ##
        from matplotlib import pyplot as plt
        for i in range(5):
            plt.imshow(all_images_ndarray[i], interpolation = "nearest")
            plt.show()
    
    del all_images
    gc.collect()

    cnt = 0
    filter_list = [None] * len(all_images_ndarray)
    
    for el in all_images_ndarray:
        shape = el.shape
        if shape[0] >= min_vals[0] and shape[1] >= min_vals[1] and el.ndim == 3:
            filter_list[cnt] = True
        else:
            filter_list[cnt] = False
        cnt += 1
        
    ## filter only those images having at least shape min_width x min_height and 3 as third dimension
    all_images_ndarray = list(compress(all_images_ndarray, filter_list))
    # transform that each image has the shape for batch training
    all_images_ndarray_resized = [resize_helper(img=img, min_vals=min_vals) for img in all_images_ndarray]
    
    del all_images_ndarray
    gc.collect()
    
    # add dimension such that we can generate a dataset
    all_images_ndarray_resized = list(map(expander, all_images_ndarray_resized))

    # vertical stack resized images in order to get dataset:
    final_images_stacked = np.vstack(all_images_ndarray_resized)

    ## if pytorch [batch_size, color_channels. height, width]
    if channels_first:
        final_images_stacked = final_images_stacked.swapaxes(1, 3).swapaxes(2, 3)
    ## elif tensorflow [batch_size, height, width, color_channels]

    print("Shape of dataset: {}".format(final_images_stacked.shape))

    del all_images_ndarray_resized
    gc.collect() 
    
    ## show first 5 images ##
    if show_image:
        from matplotlib import pyplot as plt
        for i in range(5):
            plt.imshow(final_images_stacked[i], interpolation = "nearest")
            plt.show()
    
    #res = [all_images_ndarray_resized, final_images_stacked, all_images_ndarray]
    
    return final_images_stacked

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocessed the scraped images. Preprocessing means to resize the images to a shape and\
                                                 Storing the images all together as 4-dimensional numpy array. For training the generative model")

    parser.add_argument("-fp", "--filepath", type=str, default="nude-paintings", dest="filepath",
                        help="Where the downloaded images are stored to preprocess. Default is 'nude-painting'")
    parser.add_argument("-mh", "--min_height", type=int, dest="min_height", default=128,
                        help="Minimum height value of image. Default: 128")
    parser.add_argument("-mw", "--min_width", type=int, dest="min_width", default=128,
                        help="Minimum width value of image. Default: 128")
    parser.add_argument("-cf", "--channel_first", type=bool, dest="channel_first", default=True,
                        help="If the dataset should have the shape [batch_size, channel, height, width] as done in pytorch or theano backend.\
                             For tensorflow it is [batch_size, height, width, channel]. Default: True")

    args = parser.parse_args()

    print("Python script to preprocess images and saved them as numpy array for training...")
    print(args)
    print("Preprocessing...")
    processed_images = preprocess(filepath=args.filepath,
                                  min_vals=[args.min_height, args.min_width],
                                  channels_first=args.channel_first)

    ## save
    save_path = "../data/train_data"
    print("Saving images at: {}".format(save_path))
    np.save(file=save_path, arr=processed_images)

