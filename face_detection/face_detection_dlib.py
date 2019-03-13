import cv2
import dlib
import numpy
import time
import datetime
import os
file_path = "D://img//2"

PREDICTOR_PATH = "D://ai_projects//face_detection//shape_predictor_68_face_landmarks.dat"
# 1.使用dlib自带的frontal_face_detector作为我们的人脸提取器
detector = dlib.get_frontal_face_detector()
# 2.使用官方提供的模型构建特征提取器
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class NoFaces(Exception):
    pass
camera = cv2.VideoCapture(0)
num = 0
while True and cv2.waitKey(1) == -1:

    ret, im = camera.read()
    #im1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im1 = im
    rects = detector(im1, 1)
    # 4.输出人脸数，dets的元素个数即为脸的个数
    print(len(rects))
    if len(rects) >= 1:
        print("{} faces detected".format(len(rects)))
    if len(rects) == 0:
        cv2.imshow("camera", im)
        continue
    for i in range(len(rects)):
        # 5.使用predictor进行人脸关键点识别
        landmarks = numpy.matrix([[p.x, p.y] for p in predictor(im, rects[i]).parts()])
        im = im.copy()
        # 使用enumerate 函数遍历序列中的元素以及它们的下标
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.circle(im, pos, 3, color=(0, 255, 0))
            #st = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
            #print(st+".png")
            num =num +1
            cv2.imwrite("D://img//2//"+ str(num)+".png", im)

            #
            cv2.imshow("camera", im)
camera.release()
cv2.destroyAllWindows()






file_list = os.listdir(file_path)
for cur_file in file_list :
    print(cur_file)
    im = cv2.imread(file_path + "//" +cur_file )
    im1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # 3.使用detector进行人脸检测 rects为返回的结果
    rects = detector(im1, 1)
    # 4.输出人脸数，dets的元素个数即为脸的个数
    print(len(rects))
    if len(rects) >= 1:
        print("{} faces detected".format(len(rects)))
    if len(rects) == 0:
        #raise NoFaces
        continue
    for i in range(len(rects)):
        # 5.使用predictor进行人脸关键点识别
        landmarks = numpy.matrix([[p.x, p.y] for p in predictor(im, rects[i]).parts()])
        im = im.copy()
        # 使用enumerate 函数遍历序列中的元素以及它们的下标
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.circle(im, pos, 3, color=(0, 255, 0))
    cv2.imwrite(file_path + "//_result" +cur_file,im)