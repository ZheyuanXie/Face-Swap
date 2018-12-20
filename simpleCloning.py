'''
File name: simpleCloning.py
Author: Ningshan Zhang, Zheyuan Xie
Date created: 2018-12-19
'''

import numpy as np
import cv2

COLOUR_CORRECT_BLUR_FRAC = 0.6
FEATHER_AMOUNT = 15
DILATE_KERNEL_SIZE = 10
LEFT_EYE_POINTS = list(range(19, 29))
RIGHT_EYE_POINTS = list(range(65, 75))

def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                                                im2_blur.astype(np.float64))

def simpleCloning(im1, im2, landmarks1, mask):
    kernel = np.ones((DILATE_KERNEL_SIZE,DILATE_KERNEL_SIZE),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    mask = (cv2.GaussianBlur(mask, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    mask = cv2.GaussianBlur(mask, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0).astype(np.uint8)
    im1_corr = correct_colours(im2,im1,landmarks1).astype(np.uint8)
    im_blend = (im2 * (1.0 - mask) + im1_corr * mask).astype(np.uint8)
    return im_blend
