# -*- coding: UTF-8 -*-
import numpy as np
import cPickle as pickle
import cv2
import os as os
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
from img_utils import (mls_affine_deformation, mls_affine_deformation_inv,
                       mls_similarity_deformation, mls_similarity_deformation_inv,
                       mls_rigid_deformation,mls_rigid_deformation_inv,
                       mls_rigid_deformation_inv_copy,mls_rigid_deformation_copy)

from Tkinter import *

def collinear(p0,p1,p2):
  x1=p1[0]-p0[0]
  y1=p1[1]-p0[1]
  x2=p2[0]-p1[0]
  y2=p2[1]-p1[1]
  return y1*x2-y2*x1

def drawTriangulations(image,meshPoints,ind):
  file_path='contourForWrap/data/triangulations%d.txt'%ind
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
      cv2.imwrite("triangulations.png",image)
      break

def warpMesh(orig, trans, faceNum, image):
  masked_image = np.zeros(image.shape, dtype=np.uint8)
  for t in range(faceNum):
    mask = np.zeros(image.shape, dtype=np.uint8)
    if abs(collinear(trans[t][0], trans[t][1], trans[t][2]))>20:
      cv2.fillPoly(mask, np.array([[trans[t][0], trans[t][1], trans[t][2]]], dtype=np.int32), (255, 255, 255))
      #masked_image = cv2.bitwise_or(masked_image, cv2.bitwise_and(cv2.warpAffine(image, cv2.getAffineTransform(np.float32([orig[t][0], orig[t][1], orig[t][2]]), np.float32([trans[t][0], trans[t][1], trans[t][2]])), (image.shape[1],image.shape[0])), mask))
      #image_piece=cv2.bitwise_and(cv2.warpAffine(image, cv2.getAffineTransform(np.float32([orig[t][0], orig[t][1], orig[t][2]]), np.float32([trans[t][0], trans[t][1], trans[t][2]])), (image.shape[1],image.shape[0])), mask)
      masked_image=cv2.bitwise_or(cv2.bitwise_and(cv2.warpAffine(image, cv2.getAffineTransform(np.float32([orig[t][0], orig[t][1], orig[t][2]]), np.float32([trans[t][0], trans[t][1], trans[t][2]])), (image.shape[1],image.shape[0])), mask),cv2.bitwise_and(masked_image,cv2.bitwise_not(mask)))
  return masked_image

def convertToMesh(points,ind):
  file_path='contourForWrap/data/triangulations%d.txt'%ind
  linesNum =len(open(file_path,'r',0).readlines())
  file = open(file_path,'r+',0)
  mesh=np.zeros((linesNum,3,2))
  count=0
  while True:
      num_str=file.readline()
      if num_str:
          num_array=num_str.split()
          x1=points[int(num_array[0])][int(num_array[1])][0]
          y1=points[int(num_array[0])][int(num_array[1])][1]
          x2=points[int(num_array[2])][int(num_array[3])][0]
          y2=points[int(num_array[2])][int(num_array[3])][1]
          x3=points[int(num_array[4])][int(num_array[5])][0]
          y3=points[int(num_array[4])][int(num_array[5])][1]
          mesh[count][0]=np.array([x1,y1])
          mesh[count][1]=np.array([x2,y2])
          mesh[count][2]=np.array([x3,y3])
          count=count+1
      else:
        mesh=mesh[:count,:,:]
        return mesh,count
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

def modifyModel(modify,modifyPath,s1,s2,s3,s4,s5,s6,s7):
    modify['betas'][0]=modify['betas'][0]+s1
    modify['betas'][1]=modify['betas'][1]+s2
    modify['betas'][2]=modify['betas'][2]+s3
    modify['betas'][3]=modify['betas'][3]+s4
    modify['betas'][4]=modify['betas'][4]+s5
    modify['betas'][5]=modify['betas'][5]+s6
    modify['betas'][6]=modify['betas'][6]+s7
    with open(modifyPath,'w') as f:
        pickle.dump(modify, f, pickle.HIGHEST_PROTOCOL)

def deformImg(fun,model,modifyModel,ind):
    file_path="contourForWrap/data/%d.npy"%ind  
    point=np.load(file_path)

    w, h = (274, 512)
    p = np.array([[0,0],[0,w],[w,h],[h,0]])
    q = np.array([[0,0],[0,w],[w,h],[h,0]])
    
    rt=model.cam_rt
    t=model.cam_t
    f=model.cam_f
    c=model.cam_c
    v=model
    modify_v=modifyModel
    k=np.zeros(5)
    camera_mtx=np.array([[f[0], 0, c[0]],[0., f[1], c[1]],[0.,0.,1.]], dtype=np.float64)

    image = cv2.imread(os.path.join(sys.path[0], "../../images/image"+str(ind)+".png"))
    #circle_image=image.copy()  	

    point_num=np.size(point)

    for i in range(point_num):
        v_tmp = v[point[i]]
        modify_v_tmp= modify_v[point[i]]
        #position=cv2.projectPoints(v_tmp, rt, t, camera_mtx, k)
        position=projectPoints(v_tmp, rt, t, camera_mtx, k)
        #modify_position=cv2.projectPoints(modify_v_tmp, rt, t, camera_mtx, k)
        modify_position=projectPoints(modify_v_tmp, rt, t, camera_mtx, k)
        p=np.append(p,[[int(position[0]/2),int(position[1]/2)]],axis=0)
        q=np.append(q,[[int(modify_position[0]/2),int(modify_position[1]/2)]],axis=0)
        #cv2.circle(image,(int(position[0]/2),int(position[1]/2)), 1, (0,0,0), -1)
        #cv2.circle(image,(int(modify_position[0]/2),int(modify_position[1]/2)), 1, (125,125,125), -1)

    file_path="contourForWrap/data/meshPoints%d.npy"%ind  
    points=np.load(file_path)
    vx=np.zeros((points.shape[0],points.shape[1]))
    vy=np.zeros((points.shape[0],points.shape[1]))
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            vx[i][j]=points[i][j][0]
            vy[i][j]=points[i][j][1]

    
    transformed_meshPoints=fun(image, q, p, vx, vy, alpha=1)
    transformed_meshPoints=np.transpose(transformed_meshPoints,(1,2,0))
    trans_mesh,faceNum=convertToMesh(transformed_meshPoints,ind)
    orig_mesh,faceNum=convertToMesh(points,ind)
    trans_image=warpMesh(orig_mesh,trans_mesh,faceNum,image)
    '''
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            cv2.circle(image,(int(points[i][j][0]),int(points[i][j][1])), 2, (0,255,0), -1)
            cv2.circle(image,(int(transformed_meshPoints[i][j][0]),int(transformed_meshPoints[i][j][1])), 2, (255,0,0), -1)
    cv2.imwrite("newMeshPoints.png",image)
    drawTriangulations(image,transformed_meshPoints)
    #drawTriangulations(image,points)
    '''

    #trans_image[trans_image==0]=255
    img_path="../results/result%d.png"%ind
    #cv2.imwrite(img_path,trans_image)
    cv2.imshow(img_path,trans_image)
    cv2.waitKey(0)

    '''
    #cv2.imwrite("results/circle.png",circle_image)
    img_path="results/%d.png"%ind
    print fun(image,p,q,alpha=1.0,density=0.2).astype(np.int16)
    
    cv2.imshow(img_path,fun(image,p,q,alpha=1.0,density=0.2))
    cv2.waitKey(0)
    cv2.destroyWindow()
    '''

def showModelImage(modelPath,imagePath):
    m = load_model(modelPath)

    rn = ColoredRenderer()
    ## Assign attributes to renderer
    w, h = (548, 1024)

    rn.camera = ProjectPoints(v=m, rt=np.zeros(3), t=np.array([0, 0, 2.]), f=np.array([w,w])/2., c=np.array([w,h])/2., k=np.zeros(5))
    rn.frustum={'near':1,'far':10,'width':w,'height':h}
    rn.set(v=m, f=m.f, bgcolor=np.zeros(3))

    ## Construct point light source

    rn.vc = LambertianPointLight(
        f=m.f,
        v=rn.v,
        num_verts=len(m),
        light_pos=np.array([-1000,-1000,-2000]),
        vc=np.ones_like(m)*.9,
        light_color=np.array([1., 1., 1.]))

    cv2.imwrite(imagePath,rn.r*255)

if __name__ == '__main__':
    root = Tk()
    s1 = Scale(root,
          from_ = -3,#设置最小值
          to = 3,#设置最大值
          orient = HORIZONTAL,#设置横向
          resolution=0.05,#设置步长
          tickinterval = 10,#设置刻度
          length = 600,# 设置像素
          )
    s1.pack()
    s2 = Scale(root,
          from_ = -3,#设置最小值
          to = 3,#设置最大值
          orient = HORIZONTAL,#设置横向
          resolution=0.05,#设置步长
          tickinterval = 10,#设置刻度
          length = 600,# 设置像素
          )
    s2.pack()
    s3 = Scale(root,
          from_ = -3,#设置最小值
          to = 3,#设置最大值
          orient = HORIZONTAL,#设置横向
          resolution=0.05,#设置步长
          tickinterval = 10,#设置刻度
          length = 600,# 设置像素
          )
    s3.pack()
    s4 = Scale(root,
          from_ = -3,#设置最小值
          to = 3,#设置最大值
          orient = HORIZONTAL,#设置横向
          resolution=0.05,#设置步长
          tickinterval = 10,#设置刻度
          length = 600,# 设置像素
          )
    s4.pack()
    s5 = Scale(root,
          from_ = -3,#设置最小值
          to = 3,#设置最大值
          orient = HORIZONTAL,#设置横向
          resolution=0.05,#设置步长
          tickinterval = 10,#设置刻度
          length = 600,# 设置像素
          )
    s5.pack()
    s6 = Scale(root,
          from_ = -3,#设置最小值
          to = 3,#设置最大值
          orient = HORIZONTAL,#设置横向
          resolution=0.05,#设置步长
          tickinterval = 10,#设置刻度
          length = 600,# 设置像素
          )
    s6.pack()
    s7 = Scale(root,
          from_ = -3,#设置最小值
          to = 3,#设置最大值
          orient = HORIZONTAL,#设置横向
          resolution=0.05,#设置步长
          tickinterval = 10,#设置刻度
          length = 600,# 设置像素
          )
    s7.pack()
    def wrapImage():
        ## Load SMPL model (here we load the female model)
        #pkl_paths = sorted(glob(join("pkl/","image*[0-9]_smpl.pkl")))
        pkl_paths = sorted(glob(join("pkl/","image45_smpl.pkl")))
        for ind, pkl_path in enumerate(pkl_paths):
            ind=int(re.findall(r"pkl/image(.+?)_smpl.pkl",pkl_path)[0])
            modifyPath="pkl/modify_image%d_smpl.pkl"%ind
            m = load_model(pkl_path)
            modify_r=open(pkl_path)
            modify=pickle.load(modify_r)
            modifyModel(modify,modifyPath,s1.get(),s2.get(),s3.get(),s4.get(),s5.get(),s6.get(),s7.get())
            m_modify=load_model(modifyPath)
            deformImg(mls_rigid_deformation_inv,m,m_modify,ind)
    def showModel():
        modify_r=open("pkl/image45_smpl.pkl")
        modifyPath="pkl/modify_image45_smpl.pkl"
        modify=pickle.load(modify_r)
        modifyModel(modify,modifyPath,s1.get(),s2.get(),s3.get(),s4.get(),s5.get(),s6.get(),s7.get())
        showModelImage(modifyPath,'smpl.png')
    Button(root,text = 'Wrap the image',command = wrapImage).pack()#用command回调函数获取位置
    Button(root,text = 'Show the model',command = showModel).pack()#用command回调函数获取位置
    mainloop()

## Could also use matplotlib to display
# import matplotlib.pyplot as plt
# plt.ion()
# plt.imshow(rn.r)
# plt.show()
# import pdb; pdb.set_trace()
