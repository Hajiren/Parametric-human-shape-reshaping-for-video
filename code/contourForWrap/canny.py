#-*- coding:utf-8 -*-
import numpy as np
import cv2
from ctypes import *

def findMeshPoints(img):
	grayimg = cv2.GaussianBlur(img,(3,3),0)
	canny = cv2.Canny(grayimg, 150, 200)
	cv2.imwrite("canny.png",canny)
	canny=np.nonzero(canny)

	'''
	y_min=canny[0][0]
	y_max=canny[0][canny[0].size-1]
	step=20
	points=[]
	current_index=0
	canny_temp=canny
	
	while(y_min<y_max):
		y_temp=y_min+step/2
		canny_y=np.extract(canny_temp[0]<=y_temp,canny_temp[0])
		index=canny_y.size
		canny_x=canny_temp[1][0:index-1]
		canny_temp=np.array([canny_temp[0][index:],canny_temp[1][index:]])
		y_min=y_temp	
		
		canny_x_sorted=np.sort(canny_x)
		x_min=canny_x_sorted[0]
		x_max=canny_x_sorted[-1]
		current_index_temp=0
		while(x_min<x_max):
			x_temp=x_min+step
			canny_x_temp=np.extract(canny_x<=x_temp,canny_x)
			canny_x_temp=np.extract(canny_x_temp>=x_min,canny_x_temp)
			if canny_x_temp.size!=0:
				#print canny_y_temp
				#print current_index,current_index_temp,canny_y_temp.size,current_index+current_index_temp+canny_y_temp[int(canny_y_temp.size/2)]-1
				canny_x_sort=np.argsort(canny_x_temp)
				points.append(current_index+current_index_temp+canny_x_sort[int(canny_x_sort.size/2)])
				current_index_temp=current_index_temp+canny_x_temp.size
			x_min=x_temp
		current_index=current_index+index
	for point in points[0:len(points)-1]:
		cv2.circle(img,(canny[1][point],canny[0][point]),2,(0,0,255),-1)
	'''
	size=img.shape
	step=20
	shortestDis=np.full((int(size[1]/step+1),int(size[0]/step)+1),step*step/2)
	shortestPosition=np.full((int(size[1]/step+1),int(size[0]/step+1),2),0)
	for i in range(canny[0].size):
		cage_y=int(canny[0][i]/step)
		cage_x=int(canny[1][i]/step)
		center=((cage_x+0.5)*step,(cage_y+0.5)*step)
		distance=(canny[1][i]-center[0])**2+(canny[0][i]-center[1])**2
		if(distance<shortestDis[cage_x][cage_y]):
			shortestPosition[cage_x][cage_y]=(canny[1][i],canny[0][i])
			shortestDis[cage_x][cage_y]=distance
	for x in range(int(size[1]/step+1)):
		for y in range(int(size[0]/step+1)):
			if ((shortestPosition[x][y][0]==0) & (shortestPosition[x][y][1]==0)):
				shortestPosition[x][y]=((x+0.5)*step,(y+0.5)*step)
			cv2.circle(img,(int(shortestPosition[x][y][0]),int(shortestPosition[x][y][1])),2,(0,0,255),-1)
	
	cv2.imwrite("meshPoints.png",img)
	

image = cv2.imread('../../images/image1.png')

findMeshPoints(image)
#print np.argsort(canny[1])

'''
#读取图片
img = cv2.imread('image45.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
ret, binary = cv2.threshold(gray,250,255,cv2.THRESH_BINARY)  
  
contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  

new_contours=np.reshape(contours[1],(1,contours[1].size/2,2))[0]
for contour in contours[2:]:
	contour=np.reshape(contour,(1,contour.size/2,2))[0]
	new_contours=np.append(contour,new_contours,axis=0) 
print new_contours
'''
'''
sample_contour=np.array([])
for i in range(new_contours.size/10):
	result=findMinMax(new_contours,i*5)
	print result
	if result:
		sample_contour=np.append(result[0],sample_contour)
		sample_contour=np.append(result[1],sample_contour)

print sample_contour
#print np.reshape(contour,(contour.size/2,1,2))
'''
'''
#创建白色幕布
temp = np.ones(binaryImg.shape,np.uint8)*255
#画出轮廓：temp是白色幕布，contours是轮廓，-1表示全画，然后是颜色，厚度
cv2.drawContours(temp,contours,-1,(0,255,0),1)

cv2.imshow("contours",temp)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
