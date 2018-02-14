import os

import numpy as np
import scipy.misc

from stylize import stylize

import math
from argparse import ArgumentParser

from PIL import Image
# default arguments
CONTENT_WEIGHT = 5e0
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 5e2
TV_WEIGHT = 1e2
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
ITERATIONS = 1000
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
POOLING = 'max'

def imread(path):   #读取图片
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:,:,:3]
    return img

content='examples/beili.jpg'    #此处为内容图片路径，可修改
styles=['examples/1-style.jpg']    #此处为风格图片路径，可修改

content_image = imread(content)  #读取content图片
style_images = [imread(style) for style in styles]   #读取style图片，可以有多个
initial_noiseblend = 1.0
initial = content_image
style_blend_weights = [1.0/len(style_images) for _ in style_images]
for iteration, image in stylize(
    network=VGG_PATH,
    initial=initial,
    initial_noiseblend=initial_noiseblend,
    content=content_image,
    styles=style_images,
    preserve_colors=None,
    iterations=ITERATIONS,
    content_weight=CONTENT_WEIGHT,
    content_weight_blend=CONTENT_WEIGHT_BLEND,
    style_weight=STYLE_WEIGHT,
    style_layer_weight_exp=STYLE_LAYER_WEIGHT_EXP,
    style_blend_weights=style_blend_weights,
    tv_weight=TV_WEIGHT,
    learning_rate=LEARNING_RATE,
    beta1=BETA1,
    beta2=BETA2,
    epsilon=EPSILON,
    pooling=POOLING,
    print_iterations=None,
    checkpoint_iterations=None
):
    print(iteration)
