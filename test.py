'''
Filename: test.py
Author: Ningshan Zhang, Zheyuan Xie
Date created: 2018-12-16
'''

import cv2
import numpy as np
from scipy.spatial import Delaunay
from interp import interp2
from loader import loadvideo, loadlandmarks, loadlandmarks_facepp, vislandmarks
from getLandmarks import get_landmarks
from getMask import get_face_mask
from smoothTraj import smooth_landmark_traj
from simpleCloning import simpleCloning
from seamlessCloningPoisson import seamlessCloningPoisson
from calcSimilarity import findMinDistFace, findMinDistFace_static
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    easy1 = 'Datasets/Easy/FrankUnderwood.mp4'
    easy2 = 'Datasets/Easy/MrRobot.mp4'
    easy3 = 'Datasets/Easy/JonSnow.mp4'
    medium1 = 'Datasets/Medium/LucianoRosso1.mp4'
    medium2 = 'Datasets/Medium/LucianoRosso2.mp4'
    medium3 = 'Datasets/Medium/LucianoRosso3.mp4'
    hard1 = 'Datasets/Hard/Joker.mp4'
    hard2 = 'Datasets/Hard/LeonardoDiCaprio.mp4'
    xi = 'Datasets/xidada.mp4'
    source_video_path = easy2
    target_video_path = easy1
    source_video = loadvideo(source_video_path)
    target_video = loadvideo(target_video_path)
    target_video_with_landmark = vislandmarks(target_video_path, use_facepp=True)
    source_landmarks = smooth_landmark_traj(loadlandmarks_facepp(source_video_path))
    target_landmarks = smooth_landmark_traj(loadlandmarks_facepp(target_video_path))

    # Face Matching
    # source_ind = findMinDistFace(source_landmarks, target_landmarks)
    source_ind = findMinDistFace_static(source_landmarks, target_landmarks)

    N_FRAMES = len(target_video)
    output = target_video.copy()

    time0 = time.time()
    for i in range(N_FRAMES):
        img1 = source_video[source_ind[i]].copy()
        img2 = target_video[i].copy()
        landmarks1=source_landmarks[source_ind[i]]
        landmarks2=target_landmarks[i]
        if (landmarks1 is None or landmarks2 is None):
             continue
        landmarks1 = landmarks1.astype(int)
        landmarks2 = landmarks2.astype(int)
        
        # Transform face from image1 (Frank) to align with image2 (Mr.Robot)
        T = cv2.estimateRigidTransform(landmarks1, landmarks2, False)
        if T is None:
            landmarks1_trans = landmarks1
            img1_trans = img1
        else:
            T_full = np.vstack((T,np.array([0,0,1])))
            landmarks1_full = np.vstack((landmarks1.T,np.ones((1,landmarks1.shape[0]))))
            landmarks1_trans = np.dot(T_full,landmarks1_full)
            landmarks1_trans = landmarks1_trans[0:2,:].T
            img1_trans = cv2.warpAffine(img1,T,(img2.shape[1],img2.shape[0]))
        time1 = time.time()

        # for groups in landmarks1_trans.astype(int):
        #         cv2.circle(img1_trans, (groups[0],groups[1]), 3, (0, 0, 255), 2)
        # for groups in landmarks2.astype(int):
        #         cv2.circle(img2, (groups[0],groups[1]), 3, (0, 0, 255), 2)
        # cv2.imshow('trans',img1_trans)
        # cv2.imshow('img2',img2)
        # cv2.waitKey(0)

        # Create a Delaunay triangulation
        four_corners = np.array([[0,0],[img2.shape[1]-1,img2.shape[0]-1],[0,img2.shape[0]-1],[img2.shape[1]-1,0]])
        img2_reference_points = np.vstack((landmarks2,four_corners))
        tri = Delaunay(img2_reference_points)
        mask=get_face_mask(img2,landmarks2)[:,:,0]
        position=np.where(mask>=0)
        points=np.vstack((position[1],position[0])).T
        belong_to_tri=tri.find_simplex(points)

        # find points in Delaunay triangles
        target_face = np.zeros_like(img2,dtype=np.uint8)
        for ind,triangle in enumerate(tri.simplices):
            cartesian_coor = points[np.where(belong_to_tri == ind)]
            A=np.vstack((img2_reference_points[triangle,:].T,np.array([1,1,1])))
            b=np.hstack((cartesian_coor, np.ones([cartesian_coor.shape[0],1])))
            barycentric_coor=np.linalg.inv(A).dot(b.T)
            img1_reference_points = np.vstack((landmarks1_trans,four_corners))
            interp_position = barycentric_coor.T.dot(img1_reference_points[triangle, :])
            for k in range(3):
                interp_1 = interp2(img1_trans[:, :, k], np.array([interp_position[:, 0]]), np.array([interp_position[:, 1]])).T
                target_face[:, :, k][cartesian_coor[:,1],cartesian_coor[:,0]]=interp_1.reshape(-1,)
        time2 = time.time()

        # img1_reference_points = np.vstack((landmarks1_trans,four_corners))
        # plt.figure(1)
        # plt.imshow(cv2.cvtColor(img1_trans,cv2.COLOR_RGB2BGR))
        # plt.triplot(img1_reference_points[:,0], img1_reference_points[:,1], tri.simplices.copy())
        # plt.figure(2)
        # plt.imshow(cv2.cvtColor(target_face,cv2.COLOR_RGB2BGR))
        # plt.triplot(img2_reference_points[:,0], img2_reference_points[:,1], tri.simplices.copy())
        # plt.figure(3)
        # plt.imshow(cv2.cvtColor(img2,cv2.COLOR_RGB2BGR))
        # plt.triplot(img2_reference_points[:,0], img2_reference_points[:,1], tri.simplices.copy())
        # plt.show()
        cv2.imshow('img1_trans',target_face)
        cv2.imshow('tgt',img1_trans)
        cv2.waitKey(0)
        
        target_mask = get_face_mask(target_face,landmarks2.astype(int)).astype(np.uint8)
        output[i] = seamlessCloningPoisson(target_face,img2,target_mask[:,:,0],landmarks2)
        # output[i] = simpleCloning(target_face,img2,landmarks2,target_mask)
        # output[i] = cv2.seamlessClone(img1_trans,img2,target_mask,(179,319),cv2.NORMAL_CLONE)
        time3 = time.time()

        # cv2.imshow('tagret',img2)
        # cv2.imshow('source',(target_face*target_mask).astype(np.uint8))
        cv2.imshow('output',output[i])
        cv2.waitKey(10)
        
        print("processing frame %d, transform: %.2f morph: %.2f blend %.2f total:%.2f"%(i,time1-time0,time2-time1,time3-time2,time3-time0))
        time0 = time.time()
    
    # save to file
    out = cv2.VideoWriter('result.avi',0,cv2.VideoWriter_fourcc('M','J','P','G'),20.0,(output[0].shape[1],output[0].shape[0]))
    for i in range(N_FRAMES):
        out.write(output[i])
    out.release()
    
    # playback result
    while 1:
        for i in range(N_FRAMES):
            cv2.imshow("output",output[i])
            cv2.imshow("original",target_video[i])
            cv2.imshow("landmark",target_video_with_landmark[i])
            cv2.waitKey(50)