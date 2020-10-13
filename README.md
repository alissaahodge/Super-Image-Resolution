# Super-Image-Resolution
## Enhancing Image Quality With Deep Learning

This project implements image super-resolution using convolution neural networks and auto-encoders. This project idea was taken from a coursera course and part of the implementation was guided from an article. I will link both of these.

The idea here is to train a neural network using auto encoders in order to enhance the resolution of images!
If you want to understand the code andthe process some more check main.py where I’ve left a lot of comments explaining step by step what’s being done.

## About The Code

We start by using a dataset of face images, (lfw folder), I've decreased the amount of photos in the dataset and added it to the repo. Then we lower the resolution of each and use these images as the input. The idea is to take these distorted images and feed it to our model and make model learn to get the original image back (i.e to enhance that image quality so we can get the original back).

## Setting This Up and Getting Started

1.) You can use a virtual environment if you wish
2.) run `pip install -r requirements.txt`
3.) run `python main.py <path to the image you want to test>`
Running the code and training the model etc. will take a few minutes. Please be patient and wait on the results :) !
