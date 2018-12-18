import pickle
import cv2
from getLandmarks import get_landmarks

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

def loadlandmarks(filename):
    # Load landmarks from file
    with open(filename.split('.')[0]+'.landmarks', 'rb') as f:
        landmarks = pickle.load(f)
    return landmarks

def vislandmarks(filename, play = False):
    video = loadvideo(filename)
    landmarks = loadlandmarks(filename)
    n_frames = len(video)
    for i in range(n_frames):
        if landmarks[i] is not None:
            for groups in landmarks[i]:
                cv2.circle(video[i], (groups[0],groups[1]), 3, (255, 0, 0), 2)
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
    vislandmarks(filename1)
    vislandmarks(filename2)
    vislandmarks(filename3)