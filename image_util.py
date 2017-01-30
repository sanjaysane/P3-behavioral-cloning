import numpy as np
import pandas as pd
import scipy.ndimage
import cv2
import os
import json
import errno
from PIL import Image
from scipy.stats import bernoulli



# return images with alternative brightness
def img_alter_brightness(image):
    # convert to HSV so that its easy to adjust brightness
    img1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    # alternative brightness reduction factor
    alter_bright = .25+np.random.uniform()

    # Apply the brightness reduction to the V channel
    img1[:,:,2] = img1[:,:,2]*alter_bright

    # convert to RBG again
    img1 = cv2.cvtColor(img1,cv2.COLOR_HSV2RGB)
    return img1
    
#crops image from top and bottom - inputs are factors between 0 to 0.5
def img_crop(image, top, bottom):
    assert 0 <= top < 0.5 
    assert 0 <= bottom < 0.5

    img_top = int(np.ceil(image.shape[0] * top))
    img_bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom))

    return image[img_top:img_bottom, :]

#resize img. 
def img_resize(image, new_dim):
    return scipy.misc.imresize(image, new_dim)
    
# flip image
def img_flip(image, steering_angle, flip_prob=0.5):
    do_flip = bernoulli.rvs(flip_prob)
    if do_flip:
        return np.fliplr(image), -1 * steering_angle
    else:
        return image, steering_angle
        
# gamma reduction as brightness reduction technique
# http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def img_brightness(image):
#    gamma = np.random.uniform(0.4, 1.5)
#    inv_gamma = 1.0 / gamma
#    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#    return cv2.LUT(image, table)

    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    # Apply the brightness reduction to the V channel
    image1[:,:,2] = image1[:,:,2]*random_bright
    # convert to RBG again
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1
    
# image shearing applied to it
# https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk

def img_shear(image, steering_angle, shear_range=200):
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering
    return image, steering_angle

# image rotation
def random_rotation(image, steering_angle, rotation_amount=15):
    angle = np.random.uniform(-rotation_amount, rotation_amount + 1)
    rad = (np.pi / 180.0) * angle
    return rotate(image, angle, reshape=False), steering_angle + (-1) * rad

    
