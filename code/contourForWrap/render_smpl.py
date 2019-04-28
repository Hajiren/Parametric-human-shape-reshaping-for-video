import numpy as np
import cPickle as pickle
import cv2
import os as osi
import sys
import re
import datetime
import matplotlib.pyplot as plt
from glob import glob
from os.path import join
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl_webuser.serialization import load_model
from ctypes import *

def drawTriangulations(image):
    file = open('data/triangulations.txt','r+',0)
    flag=1
    while True:
        num_str=file.readline()
        if num_str:
            if flag==1:
                x1=int(num_str)
                flag+=1
            elif flag==2:
                y1=int(num_str)
                flag+=1
            elif flag==3:
                x2=int(num_str)
                flag+=1
            elif flag==4:
                y2=int(num_str)
                flag=1
                cv2.line(image, (x1,y1), (x2, y2), (0,0,255), 1)
        else:
            cv2.imwrite("triangulations.png",image)
            break



def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def projectPoints(v,r,t,in_mtx,k):
    r_mtx=np.array([
    [np.cos(r[1])*np.cos(r[2]),np.cos(r[1])*np.sin(r[2]),-np.sin(r[1])],
    [np.cos(r[2])*np.sin(r[0])*np.sin(r[1])-np.cos(r[0])*np.sin(r[2]),np.cos(r[0])*np.cos(r[2])+np.sin(r[0])*np.sin(r[1])*np.sin(r[2]),np.cos(r[1])*np.sin(r[0])],
    [np.sin(r[0])*np.sin(r[2])+np.cos(r[0])*np.cos(r[2])*np.sin(r[1]),np.cos(r[0])*np.sin(r[1])*np.sin(r[2])-np.cos(r[2])*np.sin(r[2]),np.cos(r[0])*np.cos(r[1])] 
    ])
    out_mtx=[
    r_mtx[0][0]*float(v[0])+r_mtx[0][1]*float(v[1])+r_mtx[0][2]*float(v[2])+t[0],
    r_mtx[1][0]*float(v[0])+r_mtx[1][1]*float(v[1])+r_mtx[1][2]*float(v[2])+t[1],
    r_mtx[2][0]*float(v[0])+r_mtx[2][1]*float(v[1])+r_mtx[2][2]*float(v[2])+t[2]
    ]
    out_mtx=normalized(out_mtx,0)[0]
    '''
    out_mtx=[
    float(v[0])+t[0],
    float(v[1])+t[1],
    float(v[2])+t[2],
    ]
    '''
    x=in_mtx[0][0]*out_mtx[0]+in_mtx[0][1]*out_mtx[1]+in_mtx[0][2]*out_mtx[2]
    y=in_mtx[1][0]*out_mtx[0]+in_mtx[1][1]*out_mtx[1]+in_mtx[1][2]*out_mtx[2]

    return (x,y)

def _rotateY(points, angle):
    """Rotate the points by a specified angle."""
    ry = np.array([
        [np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ])
    return np.dot(points, ry)

def findContour(v,rt,t,camera_mtx,k,contour):
    sum_num=6889
    min_num=99999
    min_index=0
    for i in range(sum_num):
        v_tmp = v[i]
        position=projectPoints(v_tmp, rt, t, camera_mtx, k)
        distance=pow(position[0]/2-contour[0],2)+pow(position[1]/2-contour[1],2)
        if(distance<min_num):
            min_num=distance
            min_index=i
    print min_index


## Load SMPL model (here we load the female model)
#pkl_paths = sorted(glob(join("pkl/","image*[0-9]_smpl.pkl")))
pPoints=[6677,6772,6727,6585,4620,4603,4583,4582,4569,4553,4493,4507,4465,4460,4459,4335,4712,4801,4310,4298,5260,6385,4285,4954,4135,4229,4755,5364,4744,4862,5188,5167,5127,5070,5061,5422,5425,5402,3681,3682,3674,412,162,165,166,480,488,523,743,711,1239,1881,1829,1875,1506,2821,1376,1710,1620,1702,1600,1686,1939,2106,2144,2628,2636,982,1023,1034,1004,1064,1085,1468,1469,1473,1474,1134,1133,3187,3202,3328,3384,3378,3348,3225,3306,3221,3218,3218,3235,3351,3401,3332,3193,3174,1375,1157,1154,1088,1079,1369,1013,1028,956,950,964,4449,4453,4988,4441,4991,4633,4634,4542,4565,4996,4573,4591,4590,4608,6574,6575,6832,6817,6796,6616,6621]
pkl_path = "./image16_smpl.pkl"
m = load_model(pkl_path)

constrained=cdll.LoadLibrary('./constrained.so')

rt=m.cam_rt
t=m.cam_t
f=m.cam_f
c=m.cam_c
v=m
k=np.zeros(5)
image = cv2.imread("image16.png")
camera_mtx=np.array([[f[0], 0, c[0]],[0., f[1], c[1]],[0.,0.,1.]], dtype=np.float64)
flag=0
for point in pPoints:
    v_tmp=v[point]
    position=projectPoints(v_tmp, rt, t, camera_mtx, k)
    if flag>0:
        constrained.insertConstraint(int(lastposition[0]/2),int(lastposition[1]/2),int(position[0]/2),int(position[1]/2))
    else:
        flag=1
    lastposition=position
cv2.imwrite("circle.png",image)
constrained.calculateCDT()
drawTriangulations(image)

#for point in contour_points:
#    findContour(v,rt,t,camera_mtx,k,point)