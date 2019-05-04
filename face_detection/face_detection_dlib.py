import cv2
import dlib
import numpy
import time
import datetime
import os

def init_dlib():
    PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
    # 1.使用dlib自带的frontal_face_detector作为我们的人脸提取器
    global detector 
    detector = dlib.get_frontal_face_detector()
    # 2.使用官方提供的模型构建特征提取器
    global predictor
    predictor = dlib.shape_predictor(PREDICTOR_PATH)


if __name__ == '__main__':
    path_save = "./save_image"
    if os.path.exists(path_save)==False:
        os.mkdir(path_save)

    init_dlib()
    camera = cv2.VideoCapture(0)
    while True and cv2.waitKey(1) == -1:

        ret, im = camera.read()
        im1 = im
        rects = detector(im1, 1)
        if len(rects) == 0:
            cv2.imshow("camera", im)
            continue
        for i in range(len(rects)):
            # 5.使用predictor进行人脸关键点识别
            landmarks = numpy.matrix([[p.x, p.y] for p in predictor(im, rects[i]).parts()])
            im = im.copy()
            for idx, point in enumerate(landmarks):
                pos = (point[0, 0], point[0, 1])
                cv2.circle(im, pos, 3, color=(0, 255, 0))
                cv2.imshow("camera", im)
            temp_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
            cv2.imwrite(path_save+"/"+ temp_time+".jpg", im)
    camera.release()
    cv2.destroyAllWindows()
