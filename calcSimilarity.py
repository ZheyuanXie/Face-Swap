'''
File name: calcSimilarity.py
Author: Ningshan Zhang, Zheyuan Xie
Date created: 2018-12-19
'''

import cv2
import numpy as np

def calcSimilarity(landmarks1, landmarks2):
    T = cv2.estimateRigidTransform(landmarks1, landmarks2, False)
    if T is None:
        return np.inf
    T_full = np.vstack((T,np.array([0,0,1])))
    landmarks1_full = np.vstack((landmarks1.T,np.ones((1,landmarks1.shape[0]))))
    landmarks1_trans = np.dot(T_full,landmarks1_full)
    landmarks1_trans = landmarks1_trans[0:2,:].T
    dist = np.sum(np.sum((landmarks1_trans-landmarks2)**2,axis=1))
    return dist

# match a single face for all target frames
def findMinDistFace_static(landmarks1, landmarks2):
    faceind = 0
    mindist = np.inf
    for i in range(len(landmarks1)):
        if landmarks1[i] is None:
            continue
        dist = 0
        for j in range(len(landmarks2)):
            if landmarks2[j] is None:
                continue
            T = cv2.estimateRigidTransform(landmarks1[i], landmarks2[j], False)
            if T is None:
                continue
            T_full = np.vstack((T,np.array([0,0,1])))
            landmarks1_full = np.vstack((landmarks1[i].T,np.ones((1,landmarks1[i].shape[0]))))
            landmarks1_trans = np.dot(T_full,landmarks1_full)
            landmarks1_trans = landmarks1_trans[0:2,:].T
            dist = dist + calcSimilarity(landmarks1_trans,landmarks2[j])
        if dist < mindist:
            faceind = i
            mindist = dist
    return (faceind * np.ones((len(landmarks2),))).astype(int)

# match a source face for each target frames
def findMinDistFace(landmarks1, landmarks2):
    faceind = np.zeros((len(landmarks2),))
    for i in range(len(landmarks2)):
        if landmarks2[i] is None:
            continue
        mindist = np.inf
        for j in range(len(landmarks1)):
            if landmarks1[j] is None:
                continue
            T = cv2.estimateRigidTransform(landmarks1[j], landmarks2[i], False)
            if T is None:
                continue
            T_full = np.vstack((T,np.array([0,0,1])))
            landmarks1_full = np.vstack((landmarks1[j].T,np.ones((1,landmarks1[j].shape[0]))))
            landmarks1_trans = np.dot(T_full,landmarks1_full)
            landmarks1_trans = landmarks1_trans[0:2,:].T
            dist = calcSimilarity(landmarks1_trans,landmarks2[i])
            if dist < mindist:
                faceind[i] = j
                mindist = dist
    return faceind.astype(int)

if __name__ == "__main__":
    from loader import loadlandmarks_facepp, loadvideo
    import time
    easy1 = 'Datasets/Easy/FrankUnderwood.mp4'
    easy2 = 'Datasets/Easy/MrRobot.mp4'
    lm1 = loadlandmarks_facepp(easy2)
    lm2 = loadlandmarks_facepp(easy1)
    video2 = loadvideo(easy1)
    print(len(lm1))

    t0 = time.time()
    ind = findMinDistFace(lm1, lm2)
    ind_s = findMinDistFace_static(lm1,lm2)
    t1 = time.time()
    print(t1-t0)

    T = cv2.estimateRigidTransform(lm1[ind[0]], lm2[0], False)
    T_full = np.vstack((T,np.array([0,0,1])))
    landmarks1_full = np.vstack((lm1[ind[0]].T,np.ones((1,lm1[ind[0]].shape[0]))))
    landmarks1_trans = np.dot(T_full,landmarks1_full)
    landmarks1_trans = landmarks1_trans[0:2,:].T
    for groups in landmarks1_trans.astype(int):
        cv2.circle(video2[0], (groups[0],groups[1]), 1, (0, 255, 255), 2)
    for groups in lm2[0].astype(int):
        cv2.circle(video2[0], (groups[0],groups[1]), 1, (0, 0, 255), 2)
    for i in range(83):
        cv2.line(video2[0],
            (lm2[0].astype(int)[i,0],lm2[0].astype(int)[i,1]),
            (landmarks1_trans.astype(int)[i,0],landmarks1_trans.astype(int)[i,1]),(0,255,255),2)
    cv2.imshow('frame',video2[0])
    cv2.waitKey(0)