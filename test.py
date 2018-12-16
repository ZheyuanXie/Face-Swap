import cv2
import dlib
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from interp import interp2
from faceswap import get_face_mask

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces
    return np.array([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def load2video(filename1,filename2,n_frames):
    cap1 = cv2.VideoCapture(filename1)
    cap2 = cv2.VideoCapture(filename2)
    video1 = np.empty((n_frames,),dtype=np.ndarray)
    video2 = np.empty((n_frames,),dtype=np.ndarray)
    for i in range(n_frames):
        ret, video1[i] = cap1.read()
        ret, video2[i] = cap2.read()
    return video1, video2

"""
num triangle number
points all the points in pts_trans
coor points in landmarks2
"""
def barycentric(num,points,coor):
    A=np.vstack((points[num,:].T,np.array([1,1,1])))
    b=np.hstack((coor, np.ones([coor.shape[0],1])))
    bary_coff=np.linalg.inv(A).dot(b.T)
    return bary_coff


if __name__ == "__main__":
    filename1 = 'CIS581Project4PartCDatasets/Easy/FrankUnderwood.mp4'
    filename2 = 'CIS581Project4PartCDatasets/Easy/MrRobot.mp4'
    n_frames = 1#
    video1, video2 = load2video(filename1,filename2,n_frames)
    output1 = np.empty((n_frames,),dtype=np.ndarray)
    output2 = np.empty((n_frames,),dtype=np.ndarray)
    for i in range(n_frames):
        print(i)
        img1 = video1[i].copy()
        dets = detector(img1, 1)
        # shp = predictor(img1, dets[0])
        landmarks1=get_landmarks(img1)
        for groups in landmarks1:
            cv2.circle(img1, (groups[0],groups[1]), 3, (255, 0, 0), 2)
        # cv2.imshow("111",img1)
        # cv2.waitKey(1000)
        output1[i] = img1.copy()
        img2 = video2[i].copy()
        dets = detector(img2, 1)
        face=dets[0]
        # shp = predictor(img2, dets[0])
        landmarks2=get_landmarks(img2)
        for groups in landmarks2:
            cv2.circle(img2, (groups[0],groups[1]), 3, (255, 0, 0), 2)
        output2[i] = img2.copy()
        # cv2.imshow("222",img2)
        # cv2.waitKey(1000)

        T = cv2.estimateRigidTransform(landmarks1, landmarks2, False)
        # print(T)
        imtrans = cv2.warpAffine(video1[i],T,(640,360))
        T_full = np.vstack((T,np.array([0,0,1])))
        pts_full = np.vstack((landmarks1.T,np.ones((1,68))))
        pts_trans = np.dot(T_full,pts_full)
        pts_trans = pts_trans[0:2,:].T
        # print(pts_trans.shape)

        tri = Delaunay(landmarks2)
        # print(landmarks2)
        # x,y = np.meshgrid(np.arange(face.left(),face.right()),np.arange(face.top(),face.bottom()))
        
        # cv2.rectangle(img2, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 3)
        # cv2.namedWindow(f, cv2.WINDOW_AUTOSIZE)
        # cv2.imshow("image",img2)
        # cv2.waitKey(0)
        # points=(np.vstack([x.flatten(),y.flatten()])).T
        points=get_face_mask(img2,landmarks2)[:,:,0]
        # cv2.imshow("img",points)
        # cv2.waitKey(0)
        position=np.where(points>0)
        points=np.vstack((position[1],position[0])).T
        # print(points.shape)
        # print(points)
        belong_to_tri=tri.find_simplex(points)
        # print(belong_to_tri)
        max_tri=np.max(belong_to_tri)
        print("max_tri",max_tri)
        # find points in Delaunay triangles
        for j in range(max_tri+ 1):
            num = tri.simplices[j]
            # print(num)
            coor = points[np.where(belong_to_tri == j)]
            # print(coor)
            bary_coff = barycentric(num, landmarks2, coor)
            # print("bary_coor",bary_coff.shape)
            interp_position = bary_coff.T.dot(pts_trans[num, :])
            # print("interp_position",interp_position.shape)
            for k in range(3):
                interp_1 = interp2(imtrans[:, :, k], np.array([interp_position[:, 0]]), np.array([interp_position[:, 1]])).T
                # print(interp_1)
                # print(coor[:,0].shape)
                video2[i][:, :, k][coor[:,1],coor[:,0]]=interp_1.reshape(-1,)
        cv2.imshow("image1", imtrans)
        cv2.waitKey(0)
        cv2.imshow("image2",video2[i])
        cv2.waitKey(0)

        # print(tri)
        # plt.triplot(pts_trans[:,0], pts_trans[:,1], tri.simplices.copy())
        # plt.triplot(landmarks2[:,0], landmarks2[:,1], tri.simplices.copy())
        # plt.plot(landmarks2[:,0], landmarks2[:,1], 'o')
        # plt.plot(pts_trans[:,0], pts_trans[:,1], 'x')
        # plt.gca().invert_yaxis()
        # plt.show()
        

    #     for j in range(maximum+1):
    #   num=Tri.simplices[j]
    #   coor=pixels[np.where(location==j)] # pairs belong to j triangle
    #   bary_coff = barycentric(num, points,coor)

        # landmarks2t = cv2.transform(landmarks2,T)
        # print(landmarks2.shape)
    #     output1[i] = imtrans# * 0.5 + output2[i] * 0.5
    # while 1:
    #     for i in range(n_frames):
    #         cv2.imshow("video1",output1[i])
    #         cv2.imshow("video2",output2[i])
    #         cv2.waitKey(50)

    # N_frames = 10
    # frames = np.empty((1,N_frames),dtype=np.ndarray)
    # for frame_ind in range(N_frames):
    #     filename = "Videowrite_easy/"+str(frame_ind+1)+".jpg"
    #     print(filename)
    #     img = cv2.imread(filename)
    #     b, g, r = cv2.split(img)
    #     img2 = cv2.merge([r, g, b])
    #     dets = detector(img, 1)
    #     face=dets[0]
    #     shp = predictor(img, dets[0])
    #     landmarks=get_landmarks(img)
    #     for groups in landmarks:
    #         cv2.circle(img, (groups[0],groups[1]), 3, (255, 0, 0), 2)
    #     frames[0,frame_ind] = img.copy()

    # while 1:
    #     for frame_ind in range(N_frames):
    #         cv2.imshow("window",frames[0,frame_ind])
    #         cv2.waitKey(50)