import numpy as np 
import cv2
from faceswap import *
from interp import *
import dlib

"""
generate M*N log-polar coordinate
"""
def transform(e,landmarks,M=50,N=360):
    nose_pos=landmarks[30,:]
    dis=np.sum((landmarks-nose_pos)**2,axis=1)
    dis_max=np.sqrt(np.max(dis,axis=0))
    x,y=np.meshgrid(M,N)
    coor_x=x.flatten()*2*np.pi/N
    coor_y=np.exp(y.flatten()*np.log(dis_max)/M)
    interp_coor_x=nose_pos[0]+np.sin(coor_x)*coor_y
    interp_coor_y=nose_pos[1]+np.cos(coor_x)*coor_y
    coor_transform=interp2(e,interp_coor_x.reshape(M,N),interp_coor_y.reshape(M,N))
    return coor_transform,dis_max

"""
face mask with area inside contour to be np.inf
"""
def e_facemask(e,mask):
    mask2=mask[:,:,1].copy()
    e[mask2==1]=np.inf
    return e


def horcarv(coor_transform):
    [h,w]=coor_transform.shape
    cumEng=np.zeros([h,w])
    path = np.zeros([h, w])
    cumEng[:,0]=coor_transform[:,0]

    for i in range(1,w):
        up=np.hstack([np.inf,cumEng[:-1,i-1]])
        down=np.hstack([np.inf,cumEng[1:,i-1]])
        compare=np.vstack([up,cumEng[:,i-1],down])
        cumEng[:,i]=cumEng[:,i-1]+np.min(compare,axis=0)
        path[:,i]=path[:,i-1]+np.argmin(compare,axis=0)-1

    end=np.argmin(cumEng[:,-1])
    contour=[[end,w-1]]

    for i in range(w-1,0,-1):
        end=[path[end,i],i-1]
        contour.append(end)

    return np.asarray(contour)


def back_transform(contour, landmarks, dis_max,M=50,N=360):
    offset_x=landmarks[30,0]
    offset_y=landmarks[30,1]
    theta=contour[:,1]/N*2*np.pi
    pho=np.exp(contour[:,0]/M*np.log(dis_max))
    x=pho*np.sin(theta)+offset_x
    y=pho*np.cos(theta)+offset_y
    return np.hstack(x.reshape(-1,1),y.reshape(-1,1))


if __name__ == "__main__":
    img = cv2.imread("2.jpg", cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])
    dets = detector(img, 1)
    face = dets[0]
    shp = predictor(img, dets[0])
    landmarks = get_landmarks(img)
    mask = get_face_mask(img, landmarks)
    enermap=e_facemask(e,mask)
    coordinate,dis_max=transform(enermap,landmarks)
    contour=horcarv(coordinate)
    pts=back_transform(contour,landmarks,dis_max)
    for i in range(np.size(pts,axis=0)):
        cv2.circle(img, (pts[i,0],pts[i,1]), 3, (255, 0, 0), 2)
    cv2.imshow("image2",img)
    cv2.waitKey(0)





