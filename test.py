'''
Filename: test.py
Author: Ningshan Zhang, Zheyuan Xie
Date created: 2018-12-16
'''

import cv2
import numpy as np
from getLandmarks import get_landmarks
from scipy.spatial import Delaunay
from interp import interp2
from faceswap import get_face_mask, correct_colours
from loader import loadvideo, loadlandmarks, vislandmarks


"""
num: triangle number
points: all the points in landmarks1_trans
coor: points in landmarks2
"""
def barycentric(num,points,coor):
    A=np.vstack((points[num,:].T,np.array([1,1,1])))
    b=np.hstack((coor, np.ones([coor.shape[0],1])))
    bary_coff=np.linalg.inv(A).dot(b.T)
    return bary_coff

if __name__ == "__main__":
    filename1 = 'CIS581Project4PartCDatasets/Easy/FrankUnderwood.mp4'
    filename2 = 'CIS581Project4PartCDatasets/Easy/MrRobot.mp4'
    source_video = loadvideo(filename1)
    target_video = loadvideo(filename2)
    target_video_with_landmark = vislandmarks(filename2)
    source_landmarks = loadlandmarks(filename1)
    target_landmarks = loadlandmarks(filename2)

    N_FRAMES = 200
    output = np.empty((N_FRAMES,),dtype=np.ndarray)

    for i in range(N_FRAMES):
        img1 = source_video[i].copy()
        img2 = target_video[i].copy()
        landmarks1=source_landmarks[i]
        landmarks2=target_landmarks[i]

        # Transform face from image1 (Frank) to align with image2 (Mr.Robot)
        T = cv2.estimateRigidTransform(landmarks1, landmarks2, False)
        T_full = np.vstack((T,np.array([0,0,1])))
        landmarks1_full = np.vstack((landmarks1.T,np.ones((1,68))))
        landmarks1_trans = np.dot(T_full,landmarks1_full)
        landmarks1_trans = landmarks1_trans[0:2,:].T
        img1_trans = cv2.warpAffine(img1,T,(640,360))

        # correct colors
        img1_trans = correct_colours(img2,img1_trans,landmarks2).astype(np.uint8)

        # Create a Delaunay triangulation
        tri = Delaunay(landmarks2)
        mask=get_face_mask(img2,landmarks2)[:,:,0]
        position=np.where(mask>0)
        points=np.vstack((position[1],position[0])).T
        belong_to_tri=tri.find_simplex(points)
        max_tri=np.max(belong_to_tri)

        # find points in Delaunay triangles
        target_face = np.zeros_like(img2,dtype=np.uint8)
        for j in range(max_tri+ 1):
            num = tri.simplices[j]
            coor = points[np.where(belong_to_tri == j)]
            bary_coff = barycentric(num, landmarks2, coor)
            interp_position = bary_coff.T.dot(landmarks1_trans[num, :])
            for k in range(3):
                interp_1 = interp2(img1_trans[:, :, k], np.array([interp_position[:, 0]]), np.array([interp_position[:, 1]])).T
                target_face[:, :, k][coor[:,1],coor[:,0]]=interp_1.reshape(-1,)
        
        target_mask = get_face_mask(target_face,landmarks2).astype(np.uint8)
        output[i] = (img2 * (1.0 - target_mask) + target_face * target_mask).astype(np.uint8)
        # output[i] = cv2.seamlessClone(img1_trans,img2,target_mask,(179,319),cv2.NORMAL_CLONE)

        print("processing frame %d"%i)
    
    # playback result
    while 1:
        for i in range(N_FRAMES):
            cv2.imshow("output",output[i])
            cv2.imshow("original",target_video_with_landmark[i])
            cv2.waitKey(50)