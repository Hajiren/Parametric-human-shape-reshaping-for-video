import cv2

fps = 6
size = (548,1024) 
videowriter = cv2.VideoWriter("result_pkl.avi",cv2.cv.FOURCC('M','J','P','G'),fps,size)

for i in range(1,83):
    img = cv2.imread('../pkl/image%d_smpl_try.png' % i)
    if i!=68:
    	videowriter.write(img)
