# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 22:48:58 2018

@author: Tuan
"""

from dcgan import DCGAN
from vae import VAE

if __name__ == "__main__":

    print("Init DCGAN_1 model")
    dcgan_1 = DCGAN(name='DCGAN_1')
    
    print("Init DCGAN_2 model")
    dcgan_1 = DCGAN(name='DCGAN_2')
   
    print('Init VAE_1 model...')
    vae_1 =  VAE(name='VAE_1')

    print('Init VAE_2 model...')
    vae_2 =  VAE(name='VAE_2')
    
    print('Init VAE_3 model...')
    vae_3 =  VAE(name='VAE_3')
    
    print('Init VAE_4 model...')
    vae_4 =  VAE(name='VAE_4')