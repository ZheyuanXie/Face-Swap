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
import time

if __name__ == "__main__":
    easy1 = 'CIS581Project4PartCDatasets/Easy/FrankUnderwood.mp4'
    easy2 = 'CIS581Project4PartCDatasets/Easy/MrRobot.mp4'
    easy3 = 'CIS581Project4PartCDatasets/Easy/JonSnow.mp4'
    medium1 = 'CIS581Project4PartCDatasets/Medium/LucianoRosso1.mp4'
    medium2 = 'CIS581Project4PartCDatasets/Medium/LucianoRosso2.mp4'
    medium3 = 'CIS581Project4PartCDatasets/Medium/LucianoRosso3.mp4'
    hard1 = 'CIS581Project4PartCDatasets/Hard/Joker.mp4'
    hard2 = 'CIS581Project4PartCDatasets/Hard/LeonardoDiCaprio.mp4'
    xi = 'CIS581Project4PartCDatasets/xidada.mp4'
    source_video_path = easy1
    target_video_path = easy2
    source_video = loadvideo(source_video_path)
    target_video = loadvideo(target_video_path)
    target_video_with_landmark = vislandmarks(target_video_path, use_facepp=True)
    source_landmarks = smooth_landmark_traj(loadlandmarks_facepp(source_video_path))
    target_landmarks = smooth_landmark_traj(loadlandmarks_facepp(target_video_path))

    N_FRAMES = 10
    output = target_video.copy()

    time0 = time.time()
    for i in range(N_FRAMES):
        img1 = source_video[0].copy()
        img2 = target_video[i].copy()
        landmarks1=source_landmarks[0].astype(int)
        landmarks2=target_landmarks[i].astype(int)
        if (landmarks1 is None or landmarks2 is None):
             continue

        # Transform face from image1 (Frank) to align with image2 (Mr.Robot)
        T = cv2.estimateRigidTransform(landmarks1, landmarks2, False)
        if T is None:
            continue
        T_full = np.vstack((T,np.array([0,0,1])))
        landmarks1_full = np.vstack((landmarks1.T,np.ones((1,landmarks1.shape[0]))))
        landmarks1_trans = np.dot(T_full,landmarks1_full)
        landmarks1_trans = landmarks1_trans[0:2,:].T
        img1_trans = cv2.warpAffine(img1,T,(img2.shape[1],img2.shape[0]))
        time1 = time.time()

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
        
        target_mask = get_face_mask(target_face,landmarks2.astype(int)).astype(np.uint8)
        output[i] = seamlessCloningPoisson(target_face,img2,target_mask[:,:,0],landmarks2)
        # output[i] = simpleCloning(target_face,img2,landmarks2,target_mask)
        # output[i] = cv2.seamlessClone(img1_trans,img2,target_mask,(179,319),cv2.NORMAL_CLONE)
        time3 = time.time()
        
        print("processing frame %d, transform: %.2f morph: %.2f blend %.2f"%(i,time1-time0,time2-time1,time3-time2))
        time0 = time.time()
    
    # playback result
    while 1:
        for i in range(N_FRAMES):
            cv2.imshow("output",output[i])
            cv2.imshow("original",target_video[i])
            cv2.imshow("landmark",target_video_with_landmark[i])
            cv2.waitKey(50)