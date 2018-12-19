#coding=utf-8  
import cv2
import numpy as np

# # dlib 61 landmarks
# FACE_POINTS = list(range(17, 68))
# MOUTH_POINTS = list(range(48, 61))
# RIGHT_BROW_POINTS = list(range(17, 22))
# LEFT_BROW_POINTS = list(range(22, 27))
# RIGHT_EYE_POINTS = list(range(36, 42))
# LEFT_EYE_POINTS = list(range(42, 48))
# NOSE_POINTS = list(range(27, 35))
# JAW_POINTS = list(range(0, 17))

# Facepp 83 landmarks
FACE_POINTS = list(range(19, 83))
JAW_POINTS = list(range(0, 19))
LEFT_EYE_POINTS = list(range(19, 29))
LEFT_BROW_POINTS = list(range(29, 37))
MOUTH_POINTS = list(range(37, 55))
NOSE_POINTS = list(range(55, 65))
RIGHT_EYE_POINTS = list(range(65, 75))
RIGHT_BROW_POINTS = list(range(75, 83))

OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS + MOUTH_POINTS,
]

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)
    for group in OVERLAY_POINTS:
        draw_convex_hull(im,landmarks[group],color=1)
    im = np.array([im, im, im]).transpose((1, 2, 0))
    return im