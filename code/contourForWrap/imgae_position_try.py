# -*- coding: UTF-8 -*-
import cv2
from smpl_webuser.serialization import load_model
import numpy as np
import os as os
import sys
import re
from glob import glob
from os.path import join
from ctypes import *
#img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#size=img.shape
#img=cv2.resize(img,(size[1]//2,size[0]//2),cv2.INTER_LINEAR)
project_points=np.array([[0,0]])

def drawTriangulations(image,meshPoints,ind):
    file_path='data/triangulations%d.txt'%ind
    file = open(file_path,'r+',0)
    while True:
        num_str=file.readline()
        if num_str:
            num_array=num_str.split()
            x1=int(num_array[0])
            y1=int(num_array[1])
            x2=int(num_array[2])
            y2=int(num_array[3])
            x3=int(num_array[4])
            y3=int(num_array[5])
            cv2.line(image, tuple(meshPoints[x1][y1].astype(int)), tuple(meshPoints[x2][y2].astype(int)), (0,0,255), 1)
            cv2.line(image, tuple(meshPoints[x1][y1].astype(int)), tuple(meshPoints[x3][y3].astype(int)), (0,0,255), 1)
            cv2.line(image, tuple(meshPoints[x3][y3].astype(int)), tuple(meshPoints[x2][y2].astype(int)), (0,0,255), 1)
        else:
            image_path='triangulations/triangulations%d.png'%ind
            cv2.imwrite(image_path,image)
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

def findContour(contour):
    sum_num=6889
    min_num=99999
    min_index=0
    for i in range(project_points.size/2):
        position=project_points[i]
        distance=pow(position[0]/2-contour[0],2)+pow(position[1]/2-contour[1],2)
        if distance<50:
            min_index=i
            break
        if(distance<min_num):
            min_num=distance
            min_index=i
    return min_index,project_points[min_index]/2

def constructMesh(img,contours,ind):
    #circle_image=img.copy()
    grayimg = cv2.GaussianBlur(img,(3,3),0)
    canny = cv2.Canny(grayimg, 150, 200)
    canny=np.nonzero(canny)
    size=img.shape
    step=20
    shortestDis=np.full((int(size[1]/step+1),int(size[0]/step+1)),step*step/2)
    shortestPosition=np.full((int(size[1]/step+1),int(size[0]/step+1),2),0)
    constrained=cdll.LoadLibrary('./constrained.so')

    last_point=contours[0]
    for contour in contours:
        cage_x=int(contour[0]/step)
        cage_y=int(contour[1]/step)
        shortestPosition[cage_x][cage_y]=contour
        shortestDis[cage_x][cage_y]=-1
        constrained.insertConstraint(int(contour[0]),int(contour[1]),int(last_point[0]),int(last_point[1]))
        last_point=contour
    constrained.insertConstraint(int(contours[0][0]),int(contours[0][1]),int(last_point[0]),int(last_point[1]))

    for i in range(canny[0].size):
        cage_y=int(canny[0][i]/step)
        cage_x=int(canny[1][i]/step)
        center=((cage_x+0.5)*step,(cage_y+0.5)*step)
        distance=(canny[1][i]-center[0])**2+(canny[0][i]-center[1])**2
        if(distance<shortestDis[cage_x][cage_y]):
            shortestPosition[cage_x][cage_y]=(canny[1][i],canny[0][i])
            shortestDis[cage_x][cage_y]=distance

    cageSize_x=int(size[1]/step)
    cageSize_y=int(size[0]/step)
    for x in range(cageSize_x):
        for y in range(cageSize_y):
            if (np.sum(shortestPosition[x][y])==0):
                continue
            img_cage_right=img[int(shortestPosition[x][y][1]-0.5*step):int(shortestPosition[x][y][1]+0.5*step),int(shortestPosition[x][y][0]):int(shortestPosition[x][y][0]+step),:]-255
            img_cage_left=img[int(shortestPosition[x][y][1]-0.5*step):int(shortestPosition[x][y][1]+0.5*step),int(shortestPosition[x][y][0]):int(shortestPosition[x][y][0]-step),:]-255
            if (np.sum(img_cage_left)>0) & (np.sum(shortestPosition[x-1][y])==0):
                shortestPosition[x-1][y]=((x-0.5)*step,(y+0.5)*step)
                constrained.insertPoint(int(shortestPosition[x-1][y][0]),int(shortestPosition[x-1][y][1]))
                #cv2.circle(circle_image,(int(shortestPosition[x-1][y][0]),int(shortestPosition[x-1][y][1])),2,(0,0,255),-1)
            if (np.sum(img_cage_left)>0) & (np.sum(shortestPosition[x][y-1])==0):
                shortestPosition[x][y-1]=((x+0.5)*step,(y-0.5)*step)
                constrained.insertPoint(int(shortestPosition[x][y-1][0]),int(shortestPosition[x][y-1][1]))
                #cv2.circle(circle_image,(int(shortestPosition[x][y-1][0]),int(shortestPosition[x][y-1][1])),2,(0,0,255),-1)
            if (np.sum(img_cage_right)>0) & (np.sum(shortestPosition[x+1][y])==0):
                shortestPosition[x+1][y]=((x+1.5)*step,(y+0.5)*step)
                constrained.insertPoint(int(shortestPosition[x+1][y][0]),int(shortestPosition[x+1][y][1]))
                #cv2.circle(circle_image,(int(shortestPosition[x+1][y][0]),int(shortestPosition[x+1][y][1])),2,(0,0,255),-1)
            if (np.sum(img_cage_right)>0) & (np.sum(shortestPosition[x][y+1])==0):
                shortestPosition[x][y+1]=((x+0.5)*step,(y+1.5)*step)
                constrained.insertPoint(int(shortestPosition[x][y+1][0]),int(shortestPosition[x][y+1][1]))
                #cv2.circle(circle_image,(int(shortestPosition[x][y+1][0]),int(shortestPosition[x][y+1][1])),2,(0,0,255),-1)
            constrained.insertPoint(int(shortestPosition[x][y][0]),int(shortestPosition[x][y][1]))
            #cv2.circle(circle_image,(int(shortestPosition[x][y][0]),int(shortestPosition[x][y][1])),2,(0,0,255),-1)
    constrained.calculateCDT(step,ind)
    file_path='data/meshPoints%d.npy'%ind
    np.save(file_path,shortestPosition)
    constrained.clearCDT()

    #cv2.imwrite("meshPoints.png",circle_image)

    #drawTriangulations(img,shortestPosition)

def main():
    '''
    global contour_points
    cv2.namedWindow("img",cv2.WINDOW_NORMAL)  # 构建窗口
    cv2.setMouseCallback("img",onmouse,0) # 回调绑定窗口
    cv2.imshow("img",img) # 显示图像
    if cv2.waitKey()==ord("q"):
        cv2.destroyAllWindows()
        contour_points=np.delete(contour_points,0,axis=0)
        for point in contour_points:
            findContour(point)
    	cv2.imwrite("result.png",img)
    '''
    global project_points
    global constrained
    img_paths=sorted(glob(join("../../images","image*[0-9].png")))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for img_path in img_paths:
        project_points=np.array([[0,0]])
        contour_points=np.array([])
        sample_project_contours=np.array([[0,0]])

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
        ret, binary = cv2.threshold(gray,250,255,cv2.THRESH_BINARY)

        ind=int(re.findall(r"../../images/image(.+?).png",img_path)[0])
        pkl_path="../pkl/image%d_smpl_try.pkl"%ind
        m = load_model(pkl_path)
        rt=m.cam_rt
        t=m.cam_t
        f=m.cam_f
        c=m.cam_c
        v=m
        k=np.zeros(5)
        camera_mtx=np.array([[f[0], 0, c[0]],[0., f[1], c[1]],[0.,0.,1.]], dtype=np.float64)
        for i in range(6889):
            v_tmp = v[i]
            position=np.reshape(projectPoints(v_tmp, rt, t, camera_mtx, k),(1,2))
            project_points=np.append(project_points,position,axis=0) 
        project_points=project_points[1:]

        contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        counter=0
        last_point=[0,0]
        for contour in contours[1:]:
            for point in contour:
                if counter==5:
                    counter=0
                if counter==0:
                    point=point[0]
                    project_point,project_position=findContour(point)
                    contour_points=np.append(project_point,contour_points)
                    sample_project_contours=np.append(np.reshape(project_position,(1,2)),sample_project_contours,axis=0)
                    last_point=point
                counter=counter+1
        constructMesh(img,sample_project_contours[1:],ind)
        file_path="data/%d.npy"%ind
        np.save(file_path,contour_points)

if __name__=="__main__":
    main()