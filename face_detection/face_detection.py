import cv2
import os

filename = "kobe.bmp"
file_path = "D://img//2"

def detect(filename,file_name_write):
    # haarcascade_frontalface_default.xml存储在package安装的位置
    face_cascade = cv2.CascadeClassifier("C://Users//934554314//Anaconda3//Lib//site-packages//cv2//data//haarcascade_frontalface_default.xml")
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #传递参数是scaleFactor和minNeighbors,分别表示人脸检测过程中每次迭代时图像的压缩率以及每个人脸矩形保留近邻数目的最小值
    #检测结果返回人脸矩形数组
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.namedWindow("Face Detected!")
    cv2.imshow("Face Detected!", img)
    cv2.imwrite(file_path + "//"+"result_"+file_name_write, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



file_list = os.listdir(file_path)
for cur_file in file_list :
    print(cur_file)
    img = cv2.imread(file_path + "//" +cur_file )
    detect(file_path + "//" +cur_file,cur_file)
    #print(img.shape)

