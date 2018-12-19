'''
  File name: seamlessCloningPoisson.py
  Author: Ningshan Zhang, Zheyuan Xie
  Date created: 2018-12-19
'''

import numpy as np
from scipy.linalg import solve
from scipy.sparse.linalg import spsolve
from scipy import signal, sparse
import cv2
import time

COLOUR_CORRECT_BLUR_FRAC = 0.6
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

def getIndexes(mask, targetH, targetW, offsetX, offsetY):
    indexes = np.zeros((targetH,targetW)).astype('int')
    n_index = np.sum(mask)
    indexes[mask==1]=np.arange(1,n_index+1)
    return indexes

def getCoefficientMatrix(indexes):
    n_indexes = np.max(indexes)
    shift_up = signal.convolve2d(indexes,np.array([[1],[0],[0]]),'same')
    shift_down = signal.convolve2d(indexes,np.array([[0],[0],[1]]),'same')
    shift_left = signal.convolve2d(indexes,np.array([[1,0,0]]),'same')
    shift_right = signal.convolve2d(indexes,np.array([[0,0,1]]),'same')
    ij = np.array(np.vstack((np.hstack((indexes[np.logical_and(indexes>0,shift_up>0)].reshape(-1,1),shift_up[np.logical_and(indexes>0,shift_up>0)].reshape(-1,1))),
          np.hstack((indexes[np.logical_and(indexes>0,shift_down>0)].reshape(-1,1),shift_down[np.logical_and(indexes>0,shift_down>0)].reshape(-1,1))),
          np.hstack((indexes[np.logical_and(indexes>0,shift_left>0)].reshape(-1,1),shift_left[np.logical_and(indexes>0,shift_left>0)].reshape(-1,1))),
          np.hstack((indexes[np.logical_and(indexes>0,shift_right>0)].reshape(-1,1),shift_right[np.logical_and(indexes>0,shift_right>0)].reshape(-1,1)))))
        )
    n_pair = ij.shape[0]
    ij = ij - 1
    i = np.hstack((ij[:,0].T,np.arange(n_indexes)))
    j = np.hstack((ij[:,1].T,np.arange(n_indexes)))
    data = np.hstack((np.ones((1,n_pair))*-1,np.ones((1,n_indexes))*4)).reshape(-1)
    coeffA = sparse.csr_matrix((data,(i,j)))
    return coeffA

def getSolutionVect(indexes, source, target, offsetX, offsetY):
    target[indexes>0]=0;
    target_contour = signal.convolve2d(target,np.array([[0,1,0],[1,0,1],[0,1,0]]),'same')
    target_contour[indexes==0]=0

    sourceH=source.shape[0]
    sourceW=source.shape[1]
    source_indexes = indexes[0:sourceH,0:sourceW]
    source_laplacian = signal.convolve2d(source,np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]),'same')
    source_laplacian[source_indexes==0]=0

    target_contour[0:sourceH,0:sourceW] = target_contour[0:sourceH,0:sourceW] + source_laplacian
    SolVectorb=target_contour[indexes>0]

    return SolVectorb

def reconstructImg(indexes, red, green, blue, targetImg):
    red=np.clip(red,0,1)
    green=np.clip(green,0,1)
    blue=np.clip(blue,0,1)
    resultImg=targetImg.copy()
    resultImg[:,:,0][indexes>0]=red
    resultImg[:,:,1][indexes>0]=green
    resultImg[:,:,2][indexes>0]=blue
    return resultImg

def seamlessCloningPoisson(sourceImg, targetImg, mask, landmark):
    sourceImg = correct_colours(targetImg,sourceImg,landmark).astype(np.uint8)
    sourceImg = sourceImg.astype(np.double) / 255.0
    targetImg = targetImg.astype(np.double) / 255.0
    targetH,targetW=targetImg[:,:,0].shape
    indexes=getIndexes(mask, targetH,targetW,0,0)
    coeffA=getCoefficientMatrix(indexes)
    red_b=getSolutionVect(indexes,sourceImg[:,:,0],targetImg[:,:,0],0,0)
    green_b=getSolutionVect(indexes,sourceImg[:,:,1],targetImg[:,:,1],0,0)
    blue_b=getSolutionVect(indexes,sourceImg[:,:,2],targetImg[:,:,2],0,0)

    red=spsolve(coeffA,red_b)
    green=spsolve(coeffA,green_b)
    blue=spsolve(coeffA,blue_b)

    resultImg=((reconstructImg(indexes,red,green,blue,targetImg))*255).astype(np.uint8)
    return resultImg
