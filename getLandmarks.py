import dlib
import numpy as np

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) > 1:
        # raise Exception('TooManyFaces')
        return None
    if len(rects) == 0:
        # raise Exception('NoFaces')
        return None
    return np.array([[p.x, p.y] for p in predictor(im, rects[0]).parts()])