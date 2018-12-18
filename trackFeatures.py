'''
Filename: trackFeatures.py
Author: Zheyuan Xie, Ningshan Zhang, Yongyi Wang
Date created: 2018-12-18
'''

import numpy as np
from numpy.linalg import inv
import cv2
from scipy import signal
from interp import interp2

WINDOW_SIZE = 11

def estimateFeatureTranslation(startX,startY,Ix,Iy,img1,img2):
    X=startX
    Y=startY
    mesh_x,mesh_y=np.meshgrid(np.arange(WINDOW_SIZE),np.arange(WINDOW_SIZE))
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    mesh_x_flat_fix =mesh_x.flatten() + X - np.floor(WINDOW_SIZE / 2)
    mesh_y_flat_fix =mesh_y.flatten() + Y - np.floor(WINDOW_SIZE / 2)
    coor_fix = np.vstack((mesh_x_flat_fix,mesh_y_flat_fix))
    I1_value = interp2(img1_gray, coor_fix[[0],:], coor_fix[[1],:])
    Ix_value = interp2(Ix, coor_fix[[0],:], coor_fix[[1],:])
    Iy_value = interp2(Iy, coor_fix[[0],:], coor_fix[[1],:])
    I=np.vstack((Ix_value,Iy_value))
    A=I.dot(I.T)
   

    for _ in range(15):
        mesh_x_flat=mesh_x.flatten() + X - np.floor(WINDOW_SIZE / 2)
        mesh_y_flat=mesh_y.flatten() + Y - np.floor(WINDOW_SIZE / 2)
        coor=np.vstack((mesh_x_flat,mesh_y_flat))
        I2_value = interp2(img2_gray, coor[[0],:], coor[[1],:])
        Ip=(I2_value-I1_value).reshape((-1,1))
        b=-I.dot(Ip)
        solution=inv(A).dot(b)
        X += solution[0,0]
        Y += solution[1,0]
    
    return X, Y

def estimateAllTranslation(startXs,startYs,img1,img2):
    I = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
    I = cv2.GaussianBlur(I,(5,5),0.2)
    Iy, Ix = np.gradient(I.astype(float))

    startXs_flat = startXs.flatten()
    startYs_flat = startYs.flatten()
    newXs = np.full(startXs_flat.shape,-1,dtype=float)
    newYs = np.full(startYs_flat.shape,-1,dtype=float)
    for i in range(np.size(startXs)):
        if startXs_flat[i] != -1:
            newXs[i], newYs[i] = estimateFeatureTranslation(startXs_flat[i], startYs_flat[i], Ix, Iy, img1, img2)
    newXs = np.reshape(newXs, startXs.shape)
    newYs = np.reshape(newYs, startYs.shape)
    return newXs, newYs

if __name__ == "__main__":
    from loader import loadvideo, loadlandmarks
    # filename = 'CIS581Project4PartCDatasets/Easy/MrRobot.mp4'
    filename = 'CIS581Project4PartCDatasets/Easy/FrankUnderwood.mp4'
    # filename = 'CIS581Project4PartCDatasets/Easy/JonSnow.mp4'
    video = loadvideo(filename)
    result = loadvideo(filename)
    landmarks = loadlandmarks(filename)

    startXs = landmarks[0][:,[0]]
    startYs = landmarks[0][:,[1]]
    for i in range(1,len(video)):
        newXs, newYs =  estimateAllTranslation(startXs, startYs, video[i-1], video[i])
        newlandmarks = np.hstack((newXs, newYs)).astype(int)
        startXs, startYs = newXs, newYs

        # draw true landmarks
        for groups in landmarks[i]:
            cv2.circle(result[i], (groups[0],groups[1]), 3, (255, 0, 0), 2)
        # draw tracker landmarks
        for groups in newlandmarks:
            cv2.circle(result[i], (groups[0],groups[1]), 3, (0, 255, 0), 2)
        cv2.imshow('result',result[i])
        cv2.waitKey(50)