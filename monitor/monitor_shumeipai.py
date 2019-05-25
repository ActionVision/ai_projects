import cv2
import sys
from PIL import Image
import datetime
import time
import numpy as np
import threading
import os

def Catch_action_with_person(window_name, camera_idx,  path_name):
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(camera_idx)
    color = (0, 255, 0)
    num = 0
    classfier = cv2.CascadeClassifier("/home/pi/Documents/wgh/ai_projects-master/monitor/haarcascade_frontalface_alt2.xml")
    object_name = 'other'
    
    while cap.isOpened():
        #temp_time = time.strftime('%Y-%m-%d-%H-%M-%S-%f',time.localtime(time.time()))
        temp_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        object_name = 'other'
        ok, frame = cap.read() 
        if not ok:
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if num==0:
            temp_grey = grey
        diff = cv2.absdiff(grey,temp_grey)
        _m,thresh = cv2.threshold(diff,30,255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3),np.uint8)
        thresh = cv2.erode(thresh,kernel,iterations = 1)
        total = cv2.countNonZero(thresh)
        num = num +1
        if total>300:
            faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=5, minSize=(10, 10))
            if len(faceRects) > 0: 
                for faceRect in faceRects:  
                    x, y, w, h = faceRect
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    object_name = 'person'
            	    #cv2.putText(frame,temp_time,(20,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
        img_name = '%s/%s_%s.jpg' % (path_name, temp_time,object_name)
        cv2.imwrite(img_name, frame)
            
	            
            
        
        cv2.putText(frame,temp_time,(50,420),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
        cv2.putText(frame,"area:%d"%total,(50,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
        cv2.imshow(window_name, frame)
        temp_grey = grey
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def get_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ok, frame = cap.read()
        if ok == True :
            cv2.imshow('ss',frame)
            cv2.waitKey(100)
        print(ok)



if __name__ == '__main__':
    path = '/home/pi/Documents/wgh/ai_projects-master/monitor/save_image'
    Catch_action_with_person("monitor", 0,  path)
