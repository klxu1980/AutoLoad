import cv2 as cv
import numpy as np
import time


model_path = "model.yml.gz"
retval = cv.ximgproc.createStructuredEdgeDetection(model_path)
#! [Register]


cap = cv.VideoCapture('D:\\自动上卷资料\\视频\\20200712112458929 00_04_43-00_06_33.avi')

ret, frame = cap.read()
img = np.float32(frame)
#img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
img = img*(1.0/255.0)
b = retval.detectEdges(img)
cv.imshow("", b)



img = cv.imread("e:\\2022-07-21-12-45-32-717-470.jpg")
img = np.float32(img)
img = img*(1.0/255.0)
a = retval.detectEdges(img)
cv.imshow("", a)


# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
frameNum = 0 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  start = time.time()
  ret, frame = cap.read()
  #frameNum += 1
  if ret == True:   
    #tempframe = frame
    SCREEN_WIDTH = 1800
    frame = cv.resize(frame, (SCREEN_WIDTH, int(SCREEN_WIDTH * frame.shape[0] / frame.shape[1])))
    #frame = frame[1000: 2000, 500: 2000]
    #frame = cv.UMat.get(frame)
    #frame = cv.UMat(frame)
    #cv.imshow('frame', frame)
    img = np.float32(frame)
    img = img*(1.0/255.0)
    a = retval.detectEdges(img)
    #orimap = retval.computeOrientation(img)
    #a = retval.edgesNms(img, orimap)
    #a = retval.computeOrientation(img)
    end = time.time()
    print(end-start)
    cv.imshow('a', a)
    cv.waitKey(1)

cap.release()
 
# Closes all the frames
cv.destroyAllWindows()
 

