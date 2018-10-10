#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title data_preprocess.py
@author: Tuan Le
@email: tuanle@hotmail.de
This script loads images in the folder ../data/genre and preprocess them into a corresponding numpy ndarray
"""

import numpy as np
from skimage import io
from skimage.transform import resize
import os
genre = "yakusha-e"
os.chdir("../data/" + genre + "/")

def preprocess():
    #list all images in ../data/'genre'/img1.jpg
    all_images = [x for x in os.listdir() if x.endswith(".jpg") | x.endswith(".png") | x.endswith(".jpeg")]
    all_images_ndarray = list(map(io.imread, all_images))
        
    #get min values in dimensions for resizing:
    #min_vals = min(list(map(np.shape, all_images_ndarray)))
    ## hardcoded: shrink size to 256x256 by hand because of latter training
    min_vals = [256, 256]    
    def resize_helper(img):
        return resize(image = img, output_shape = (min_vals[0], min_vals[1]))
        
    all_images_ndarray_resized = list(map(resize_helper, all_images_ndarray))
    # expand one dimension for later concatenation
    def expander(x):
        return np.expand_dims(x, axis=0)
        
    all_images_ndarray_resized = list(map(expander, all_images_ndarray_resized))
    final_images_stacked = np.vstack(all_images_ndarray_resized)
    
    ## show first 5 images ##
    #from matplotlib import pyplot as plt
    #for i in range(5):
    #    plt.imshow(final_images_stacked[i], interpolation = "nearest")
    #    plt.show()
    
    res = [all_images_ndarray_resized, final_images_stacked, all_images_ndarray]
    
    return res

