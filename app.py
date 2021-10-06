import cv2
import pdb
import numpy as np
#from cluster import cluster

cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	_,frame = cv2.threshold(frame, 128, 255, cv2.THRESH_BINARY)
	cv2.imshow('Input', frame)
	#pdb.set_trace()
	#clusters = cluster(cv2.pyrDown(frame))
	c = cv2.waitKey(1)
	if c == 1000:
		break