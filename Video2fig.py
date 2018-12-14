import numpy as np
import cv2
from scipy import signal
from PIL import Image


cap = cv2.VideoCapture('CIS581Project4PartCDatasets/Easy/FrankUnderwood.mp4')
# print(cap.isOpened())
frame_count=1
ret=True

while(ret):
    ret, frame = cap.read()
    print('Read a frame:', ret)
    cv2.imwrite("Videowrite_easy/" + "%d.jpg" % frame_count, frame)

    frame_count = frame_count + 1

cap.release()