import cv2

fps = 24  
size = (274,512) 
videowriter = cv2.VideoWriter("result.avi",cv2.cv.FOURCC('M','J','P','G'),fps,size)

for i in range(1,150):
    img = cv2.imread('../results/result%d.png' % i)
    if i!=70:
    	videowriter.write(img)
