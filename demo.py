'''
Filename: demo.py
Author: Zheyuan Xie, Ningshan Zhang
Date created: 2018-12-20
'''

import cv2
import numpy as np
from scipy.spatial import Delaunay
from interp import interp2
from loader import loadvideo, loadlandmarks_facepp, vislandmarks
from getLandmarks import get_landmarks
from getMask import get_face_mask
from smoothTraj import smooth_landmark_traj
from simpleCloning import simpleCloning
from seamlessCloningPoisson import seamlessCloningPoisson
from calcSimilarity import findMinDistFace_static

easy1 = 'Datasets/Easy/FrankUnderwood.mp4'
easy2 = 'Datasets/Easy/MrRobot.mp4'
easy3 = 'Datasets/Easy/JonSnow.mp4'
medium1 = 'Datasets/Medium/LucianoRosso1.mp4'
medium2 = 'Datasets/Medium/LucianoRosso2.mp4'
medium3 = 'Datasets/Medium/LucianoRosso3.mp4'
hard1 = 'Datasets/Hard/Joker.mp4'
hard2 = 'Datasets/Hard/LeonardoDiCaprio.mp4'
xidada = 'Datasets/xidada.mp4'
SOURCE_VIDEO_PATH = easy1
TARGET_VIDEO_PATH = easy2

if __name__ == "__main__":
    print("Loading videos...")
    source_video = loadvideo(SOURCE_VIDEO_PATH)
    target_video = loadvideo(TARGET_VIDEO_PATH)
    print("Loading landmarks...")
    source_landmarks = smooth_landmark_traj(loadlandmarks_facepp(SOURCE_VIDEO_PATH))
    target_landmarks = smooth_landmark_traj(loadlandmarks_facepp(TARGET_VIDEO_PATH))

    # Face Matching
    print("Matching faces...")
    source_ind = findMinDistFace_static(source_landmarks, target_landmarks)


    # For each frame in target video
    N_FRAMES = len(target_video)
    output = target_video.copy()
    for i in range(N_FRAMES):
        img1 = source_video[source_ind[i]].copy()
        img2 = target_video[i].copy()
        landmarks1=source_landmarks[source_ind[i]]
        landmarks2=target_landmarks[i]
        if (landmarks1 is None or landmarks2 is None):
             continue
        landmarks1 = landmarks1.astype(int)
        landmarks2 = landmarks2.astype(int)

        print("Frame %d/%d: warping..."%(i,N_FRAMES))

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

        print("Frame %d/%d: blending..."%(i,N_FRAMES))

        # Blend source face to target image
        target_mask = get_face_mask(target_face,landmarks2.astype(int)).astype(np.uint8)
        output[i] = seamlessCloningPoisson(target_face,img2,target_mask[:,:,0],landmarks2)

        # display frame result
        cv2.imshow('output',output[i])
        cv2.waitKey(10)
    
    # save result video to file
    out = cv2.VideoWriter('result.avi',0,cv2.VideoWriter_fourcc('M','J','P','G'),20.0,(output[0].shape[1],output[0].shape[0]))
    for i in range(N_FRAMES):
        out.write(output[i])
    out.release()
    
    # playback result video in a loop
    while 1:
        for i in range(N_FRAMES):
            cv2.imshow("output",output[i])
            cv2.imshow("original",target_video[i])
            cv2.waitKey(20)
