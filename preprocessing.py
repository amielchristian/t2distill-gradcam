from __future__ import division, print_function
import os
from multiprocessing.pool import Pool
import click
import numpy as np
from PIL import Image, ImageFilter
import data

mean = np.array([108.64628601, 75.86886597, 54.34005737]) / 255.0
std = np.array([70.74596827, 51.6363773, 43.4604063]) / 255.0

def crop(image):
    blurred = image.filter(ImageFilter.BLUR)
    ba = np.array(blurred)
    h, w, _ = ba.shape

    if w > 1.2 * h:
        left_max = ba[:, : w // 32, :].max(axis=(0, 1)).astype(int)
        right_max = ba[:, - w // 32:, :].max(axis=(0, 1)).astype(int)
        max_bg = np.maximum(left_max, right_max)

        foreground = (ba > max_bg + 10).astype(np.uint8)
        bbox = Image.fromarray(foreground).getbbox()

        if bbox is None:
            print('No bounding box')
        else:
            left, upper, right, lower = bbox
            if right - left < 0.8 * h or lower - upper < 0.8 * h:
                print('Bounding box too small')
                bbox = None
    else:
        bbox = None

    if bbox is None:
        w, h = img.size
        left = max((w - h) // 2, 0)
        upper = 0
        right = min(w - (w - h) // 2, w)
        lower = h
        bbox = square_bbox(left, upper, right, lower)

        cropped = image.crop(bbox)
    
    return cropped

def square_bbox(img):
    w, h = img.size
    left = max((w - h) // 2, 0)
    upper = 0
    right = min(w - (w - h) // 2, w)
    lower = h
    return (left, upper, right, lower)


def convert_square(fname, crop_size):
    img = Image.open(fname)
    bbox = square_bbox(img)
    cropped = img.crop(bbox)
    resized = cropped.resize([crop_size, crop_size])
    return resized


def get_convert_fname(fname, extension, directory, convert_directory):
    return fname.replace('jpeg', extension).replace(directory, 
                                                    convert_directory)

def process(args):
    fun, arg = args
    directory, convert_directory, fname, crop_size, extension = arg
    convert_fname = get_convert_fname(fname, extension, directory, 
                                      convert_directory)
    if not os.path.exists(convert_fname):
        img = fun(fname, crop_size)
        save(img, convert_fname) 

def resize(image, size):
    return image.resize([size, size])

def rescale(image):
	return image / 255.0

def z_score_normalization(image, mean, std):
    normalized_image = (image - mean) / (std + 1e-7)
    return normalized_image

def save(img, fname):
    img.save(fname, quality=97)