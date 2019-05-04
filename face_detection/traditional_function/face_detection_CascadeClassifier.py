import cv2
import os
import numpy as np

def detect(filename,file_name_write):
    try :
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
        img = cv2.imread(filename)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5, minSize=(10, 10))
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.namedWindow("Face")
        cv2.imshow("Face", img)
        #cv2.imwrite(file_path+"result_"+file_name_write, img)
        cv2.waitKey(0)
    except :
        print("error")    

file_path = "D:\\ai_projects\\face_detection\\save_image\\"

if __name__ == '__main__':

    file_list = os.listdir(file_path)
    for cur_file in file_list :
        print(cur_file)
        detect(file_path +cur_file,cur_file)