import cv2
import sys
from PIL import Image
import datetime
import time
import numpy as np

def Catch_person(window_name, camera_idx, catch_pic_num, path_name):
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(camera_idx)
    classfier = cv2.CascadeClassifier("C://Users//934554314//Anaconda3//Lib//site-packages//cv2//data//haarcascade_frontalface_alt2.xml")
    color = (0, 255, 0)
    num = 0
    while cap.isOpened():
        ok, frame = cap.read() 
        if not ok:
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=5, minSize=(10, 10))
        if len(faceRects) > 0: 
            for faceRect in faceRects:  
                x, y, w, h = faceRect
                #temp_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                temp_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
                img_name = '%s/%s.jpg' % (path_name, temp_time)
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.imwrite(img_name, frame)
                cv2.imshow(window_name, frame)
                continue
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(20)
        if c & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def Catch_action(window_name, camera_idx, catch_pic_num, path_name):
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(camera_idx)
    color = (0, 255, 0)
    num = 0
    while cap.isOpened():
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
        print(total)
        num = num +1
        if total>1000:
            cv2.putText(frame,'action:',(100,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),10)
            temp_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
            img_name = '%s/%s.jpg' % (path_name, temp_time)
            cv2.imwrite(img_name, frame)
        cv2.imshow(window_name, frame)
        cv2.imshow("before", temp_grey)
        cv2.imshow("diff", thresh)
        temp_grey = grey
        c = cv2.waitKey(20)
        if c & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def Catch_action_with_person(window_name, camera_idx, catch_pic_num, path_name):
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(camera_idx)
    color = (0, 255, 0)
    num = 0
    classfier = cv2.CascadeClassifier("C://Users//934554314//Anaconda3//Lib//site-packages//cv2//data//haarcascade_frontalface_alt2.xml")
    object_name = 'other'
    while cap.isOpened():
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
        print(total)
        num = num +1
        if total>1000:
            faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=5, minSize=(10, 10))
            if len(faceRects) > 0: 
                for faceRect in faceRects:  
                    x, y, w, h = faceRect
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    object_name = 'person'
            cv2.putText(frame,'action:',(100,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),10)
            temp_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
            img_name = '%s/%s_%s.jpg' % (path_name, temp_time,object_name)
            cv2.imwrite(img_name, frame)
        cv2.imshow(window_name, frame)
        cv2.imshow("before", temp_grey)
        cv2.imshow("diff", thresh)
        temp_grey = grey
        c = cv2.waitKey(20)
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
    #get_camera()
    Catch_action_with_person("face", 0, 1000, 'D:\\ai_projects\\monitor')
