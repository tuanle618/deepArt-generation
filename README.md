# deepArt-generation
Repository for generating art using deep convolutional generative adversarial networks where training data is scraped from wikiart.

## Motivation
The motivation behind this small project is to explore the capabilities of generative adversarial networks in order to create new (fake) artificial images based on historical/available image data. In order to get to know the idea behind GANs have a look at this [blogpost](https://www.analyticsvidhya.com/blog/2017/06/introductory-generative-adversarial-networks-gans/).  
In contrast to [Variational Autoencoders](https://sergioskar.github.io/Autoencoder/) the generative adversarial network has a more sophisticated architecture where a generator (who will create fake images) and discriminator (who will decide whether an image is fake or not) are trained *simultaneously* in a **stacked/adversarial** network optimizing some [min-max-criterion](https://medium.com/@jonathan_hui/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b):  
![Image of optimization problem](https://cdn-images-1.medium.com/max/1000/1*ihK3whUAZ_0UeK4SJicYFw.png)

## Structure ##
The repository-folder is structured onto 3 folders named `data`, `source` and  `model`.
- In `data` the scraped training images are stored.
- In `source` all python-scripts are stored. There are 3 following scripts:

  `data_scraping.py`: this script fetches images from a specific genre available on [wikiart](https://www.wikiart.org/en/artists-by-genre). Currently the genre scraped is **yakusha-e**.  
  
  `data_preprocess.py`: this script executes some preprocessing steps like merging all images into one `numpy.ndarray` and scaling all images onto one shape, such that the deep learning algorithm can be trained in the next step.  
  
  `train_model.py`: this script creates a [deep convolutional generative adversarial network](https://medium.com/@jonathan_hui/gan-dcgan-deep-convolutional-generative-adversarial-networks-df855c438f) and trains the stacked model over **115 images**, **4600 epochs** and a **batch size of 10**.
- In `model` the weights for generator and discriminator networks are saved. The `train_model.py` saves every 200 epochs the weights and 4 images of the current generator model.  
Hence, the `model` folder also contains the development of how the generator is trained with respect to its weights, such that it *should* generate images which resemble the original training images. 
  
## Data 
Data is scraped from [wikiart](https://www.wikiart.org/en/) in the python-script [data_scraping.py](https://github.com/ptl93/deepArt-generation/blob/master/source/data_scraping.py). In this version the genre **yakusha-e** is selected as genre, since I was/am interested into asian arts and there are not so many images available in the wikiart database. If you want to scrape another genre, simply change the string variable in line 20 `genre_to_scrape = "yakusha-e"` to any other genre of interest.
After scraping the yakusha-e images 115 images were stored onto my local machine.  
Here are some scraped image (in original size):  
![Image 1](https://github.com/ptl93/deepArt-generation/blob/master/data/yakusha-e/natori-shunsen_7.jpg)
![Image 2](https://github.com/ptl93/deepArt-generation/blob/master/data/yakusha-e/utagawa-toyokuni_10.jpg)
![Image 3](https://github.com/ptl93/deepArt-generation/blob/master/data/yakusha-e/yamamura-toyonari_0.jpg)
![Image 4](https://github.com/ptl93/deepArt-generation/blob/master/data/yakusha-e/yamamura-toyonari_10.jpg)

## Generated Images  
The generated images **unfortunately** do not really resemble the yakusha-e genre, more an *abstract* version of it. I think possible reasons might be the architecture of the generator network. One might add more convolutional layers, such that we can get more realistic yakusha-e images. In addition the database is rather small with 115 images. Further image-gathering can be done.  
Find below the generated images over the epochs: `[0, 200, 1200, 2200, 3400 , 4400]`:  


**Epoch 0 (so technically just noise):**  
![epoch0](https://github.com/ptl93/deepArt-generation/blob/master/model/images/dcgan_0.jpg)  
**Epoch 200**:  
![epoch200](https://github.com/ptl93/deepArt-generation/blob/master/model/images/dcgan_200.jpg)  
**Epoch 1200**:  
![epoch1200](https://github.com/ptl93/deepArt-generation/blob/master/model/images/dcgan_1200.jpg)  
**Epoch 2200**:  
![epoch2200](https://github.com/ptl93/deepArt-generation/blob/master/model/images/dcgan_2200.jpg)  
**Epoch 3400**:  
![epoch3400](https://github.com/ptl93/deepArt-generation/blob/master/model/images/dcgan_3400.jpg)  
**Epoch 4400**:  
![epoch4400](https://github.com/ptl93/deepArt-generation/blob/master/model/images/dcgan_4400.jpg)    
  
  From epoch 2200 on one can see that some structure from yakusha-e is captured in the generated images. It's up to you, to change some setting :) I somehow got memory issues when trying to increase the number of epochs and/or the batch_size. This might change the training as well...
  
## Environment & Execution   
The code was executed using `Python 3.6` and for deep learning `keras`-framework with `tensorflow`-backend.  
Note that I ran the `train_model.py` script with local-GPU using **NVIDIA GeForce GTX 1050 Ti**. When runing this script with CPU-tensoflow, execution time might be longer. On GPU the training time took about *1.5 hours*.
  
This paragraph describes how to set up your environment locally. Make you have python-module `virtualenv` installed.  
If you do not wish to run the training script on a virtual environment, make sure you have all modules/libraries properly installed and start from step on.  

Step 1 - Clone this repo:  
```git 
git clone https://github.com/ptl93/deepArt-generation
cd deepArt-generation
```  

Step 2 - create and activate a python virtual environment:
``` 
virtualenv deepArt-venv
source deepArt-venv/bin/activate
```    

Step 2 (alternative) - create a conda environment:
```
conda create --name deepArt-venv python=3.6
source activate deepArt-venv
```   

Step 3 - install needed modules/libraries:
```
pip install -r requirements.txt
```
This will download and install bs4, keras, numpy, tensorflow, scikit-learn and some other dependencies.  

Step 4 - run the python scripts:  
If you want to scrape another genre, make sure you changed the genre in codeline 20:
```python
python data_scraping.py
```
and finally train the adversarial network:
``` python
python train_model.py
``` 

## Architecture of generator and discriminator
Both, generator and discriminator networks, are convolutional neural networks with 4 layer.
The generator takes as input a 100-dimensional random-noise vector, each entry sampled from a standard gaussian distribution.
Since the job of the generator is to create an artificial (fake) image, the output is color image (256x256x3). In this case the height and width is both 256, done in `data_preprocess.py` script.  Find below the architecture of the generator:
<img src="https://github.com/ptl93/deepArt-generation/blob/master/model/generator_model.png"/>
  
The discriminator takes as input a color image (256x256x3) and outputs a probability (in range 0 to 1) whether the given image is real or fake. Find below the architecture of the discriminator:   
<img src="https://github.com/ptl93/deepArt-generation/blob/master/model/discriminator_model.png"/>  

## License
Code under MIT license.
