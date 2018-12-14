#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title data_preprocess.py
@author: Tuan Le
@email: tuanle@hotmail.de
This script loads images in the folder ../data/genre_or_style and preprocess them into a corresponding numpy ndarray
"""

import numpy as np
from skimage import io
from skimage.transform import resize
import os
import gc
from itertools import compress

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
    
def preprocess(genre_or_style, min_vals = [128,128]):
    """
    loads images from a path, resizes them to a certain range and finally stacks them
    """
    
    path = "../data/{}/".format(genre_or_style)
    #list all images in ../data/'genre_or_style'/img1.jpg
    
    all_images = [x for x in os.listdir(path) if x.endswith(".jpg") | x.endswith(".png") | x.endswith(".jpeg")]
    all_images = [os.path.join(path, image) for image in all_images]
    all_images_ndarray = list(map(io.imread, all_images))
    
    show_image = False
    
    if show_image:
        ## show first 5 images ##
        from matplotlib import pyplot as plt
        for i in range(5):
            plt.imshow(all_images_ndarray[i], interpolation = "nearest")
            plt.show()
    
    del all_images
    gc.collect()
    
    #get min values in dimensions for resizing:
    #min_vals = min(list(map(np.shape, all_images_ndarray)))
    ## hardcoded: shrink size to 128x128 by hand because of latter training
    
    ## possible enhancement: take only image which are at leat 128x128 ? if smaller and "upsizing"
    ## might lead to bad result?! 
    
    cnt = 0
    filter_list = [None] * len(all_images_ndarray)
    
    for el in all_images_ndarray:
        shape = el.shape
        if shape[0] >= min_vals[0] and shape[1] >= min_vals[1] and shape[2] == 3:
            filter_list[cnt] = True
        else:
            filter_list[cnt] = False
        cnt += 1
        
    ## filter only those images having at least shape 256x256 and 3 as third dimension
    all_images_ndarray = list(compress(all_images_ndarray, filter_list))
    # transform that each image has the shape (256,256,3)
    all_images_ndarray_resized = list(map(resize_helper, all_images_ndarray))
    
    del all_images_ndarray
    gc.collect()
    
    # add dimension such that we can generate a dataset, hence (1,256,256,3)
    all_images_ndarray_resized = list(map(expander, all_images_ndarray_resized))
    
    # vertical stack resized images in order to get dataset:
    final_images_stacked = np.vstack(all_images_ndarray_resized)
    
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