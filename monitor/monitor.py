import cv2
import sys
from PIL import Image
import datetime
import time
import numpy as np
from bypy import ByPy
import threading
import os


def upload_file(localpath,remotepath):
    bp.upload(localpath,remotepath,ondup='newcopy')

def upload_file_rename(localpath,remotepath):
    bp.upload(localpath,remotepath,ondup='newcopy')
    print("upload ok:"+localpath)
    foldername=localpath.split('\\')[-1]
    path_name = localpath.split('\\')[:-1]
    newfile =""
    for f in path_name:
        newfile = newfile+f+"\\"
    newfile = newfile +"up_"+foldername
    os.rename(localpath,newfile)
    print("rename ok:"+newfile)

def file_name(user_dir):
    file_list = list()
    for root, dirs, files in os.walk(user_dir):
        for file in files:
            if file.startswith("up_")==False:
                file_list.append(os.path.join(root, file))
    return file_list

class myThread (threading.Thread):   #继承父类threading.Thread
    def __init__(self, threadID, name, counter,file_path):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.file_path = file_path
    def run(self):                   #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数 
        temp_date = time.strftime('%Y-%m-%d',time.localtime(time.time()))
        #fobj = open("%s.txt"%temp_date,"a")
        while True:
            time.sleep(3)
            file_list = ''
            file_list = file_name(self.file_path)
            now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
            print(now_time+"\n")
            now_date = time.strftime('%Y-%m-%d',time.localtime(time.time()))
            fobj = open("%s.txt"%temp_date,"a")
            if now_date != temp_date:
                fobj = open("%s.txt"%temp_date,"a")
                print("new day")
                fobj.writelines("new day")
                fobj.flush()
            for file in file_list:
                upload_file_rename(file,dir_name)
                fobj.writelines("upload ok:%s  %s\n"%(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f'),file))
                fobj.flush()
            fobj.close()


def Catch_person(window_name, camera_idx, catch_pic_num, path_name):
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(camera_idx)
    classfier = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
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
            #print("img_name:%s..............."%img_name)
            
        
        cv2.putText(frame,temp_time,(50,420),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
        cv2.putText(frame,"area:%d"%total,(50,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
        cv2.imshow(window_name, frame)
        temp_grey = grey
        c = cv2.waitKey(1000)
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
    
    dir_name = "monitor"
    bp = ByPy()
    bp.mkdir(remotepath = dir_name)
    # 创建新线程
    thread1 = myThread(1, "Thread-1", 1,"D:\\ai_projects\\monitor\\save_image")

    # 开启线程
    thread1.start()
    Catch_action_with_person("monitor", 0, 1000, 'D:\\ai_projects\\monitor\\save_image')
