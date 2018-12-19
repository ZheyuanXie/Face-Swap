'''
Filename: loader.py
Author: Zheyuan Xie
Date created: 2018-12-18
'''

import pickle
import cv2
from getLandmarks import get_landmarks_facepp, get_landmarks

def loadvideo(filename):
    cap = cv2.VideoCapture(filename)
    images = []
    while(True):
        ret, frame = cap.read()
        if ret == False:
            break
        images.append(frame)
    cap.release()
    return images

def savelandmarks(filename):
    video = loadvideo(filename)
    landmarks = []
    print('Number of Frames: %d'%(len(video)))
    cnt = 0
    for image in video:
        landmarks.append(get_landmarks(image))
        cnt  = cnt + 1
        print('%f%%'%(cnt/len(video)*100))
    # Dump landmarks to file
    with open(filename.split('.')[0]+'.landmarks', 'wb') as f:
        pickle.dump(landmarks, f)

def savelandmarks_facepp(filename):
    video = loadvideo(filename)
    landmarks = []
    print('Number of Frames: %d'%(len(video)))
    cnt = 0
    for image in video:
        landmarks.append(get_landmarks_facepp(image))
        cnt  = cnt + 1
        print('%f%%'%(cnt/len(video)*100))
    # Dump landmarks to file
    with open(filename.split('.')[0]+'.facepp', 'wb') as f:
        pickle.dump(landmarks, f)

def loadlandmarks(filename):
    # Load landmarks from file
    with open(filename.split('.')[0]+'.landmarks', 'rb') as f:
        landmarks = pickle.load(f)
    return landmarks

def loadlandmarks_facepp(filename):
    # Load landmarks from file
    with open(filename.split('.')[0]+'.facepp', 'rb') as f:
        landmarks = pickle.load(f)
    return landmarks

def vislandmarks(filename, play = False, use_facepp = False):
    video = loadvideo(filename)
    if use_facepp:
        landmarks_facepp = loadlandmarks_facepp(filename)
    landmarks = loadlandmarks(filename)
    n_frames = len(video)
    for i in range(n_frames):
        if landmarks[i] is not None:
            for groups in landmarks[i].astype(int):
                cv2.circle(video[i], (groups[0],groups[1]), 3, (255, 0, 0), 2)
        if use_facepp:
            for groups in landmarks_facepp[i].astype(int):
                cv2.circle(video[i], (groups[0],groups[1]), 3, (0, 255, 0), 2)
        if play:
            cv2.imshow('Landmarks',video[i])
            cv2.waitKey(50)
    return video

if __name__ == "__main__":
    filename1 = 'CIS581Project4PartCDatasets/Easy/FrankUnderwood.mp4'
    filename2 = 'CIS581Project4PartCDatasets/Easy/MrRobot.mp4'
    filename3 = 'CIS581Project4PartCDatasets/Easy/JonSnow.mp4'
    # savelandmarks(filename1)
    # savelandmarks(filename2)
    # savelandmarks(filename3)
    # vislandmarks(filename1, play=True)
    # vislandmarks(filename2, play=True, use_facepp=True)
    vislandmarks(filename3, play=True, use_facepp=True)