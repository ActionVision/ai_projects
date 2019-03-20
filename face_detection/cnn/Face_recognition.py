# -*- coding: utf-8 -*-

import cv2
import sys
import gc
import random

import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K

from cnn.load_dataset import load_dataset, resize_image, IMAGE_SIZE

class Model:
    def __init__(self):
        self.model = None
        # 建立模型
    def build_model(self, dataset, nb_classes=4):
        # 构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
        self.model = Sequential()
        # 以下代码将顺序添加CNN网络需要的各层，一个add就是一个网络层
        self.model.add(Convolution2D(32, 3, 3, border_mode='same',
                                     input_shape=dataset.input_shape))  # 1 2维卷积层
        self.model.add(Activation('relu'))  # 2 激活函数层
        self.model.add(Convolution2D(32, 3, 3))  # 3 2维卷积层
        self.model.add(Activation('relu'))  # 4 激活函数层
        self.model.add(MaxPooling2D(pool_size=(2, 2)))  # 5 池化层
        self.model.add(Dropout(0.25))  # 6 Dropout层
        self.model.add(Convolution2D(64, 3, 3, border_mode='same'))  # 7  2维卷积层
        self.model.add(Activation('relu'))  # 8  激活函数层
        self.model.add(Convolution2D(64, 3, 3))  # 9  2维卷积层
        self.model.add(Activation('relu'))  # 10 激活函数层
        self.model.add(MaxPooling2D(pool_size=(2, 2)))  # 11 池化层
        self.model.add(Dropout(0.25))  # 12 Dropout层
        self.model.add(Flatten())  # 13 Flatten层
        self.model.add(Dense(512))  # 14 Dense层,又被称作全连接层
        self.model.add(Activation('relu'))  # 15 激活函数层
        self.model.add(Dropout(0.5))  # 16 Dropout层
        self.model.add(Dense(nb_classes))  # 17 Dense层
        self.model.add(Activation('softmax'))  # 18 分类层，输出最终结果
        # 输出模型概况
        self.model.summary()
    # 训练模型
    def train(self, dataset, batch_size=20, nb_epoch=10, data_augmentation=True):
        sgd = SGD(lr=0.01, decay=1e-6,
                  momentum=0.9, nesterov=True)  # 采用SGD+momentum的优化器进行训练，首先生成一个优化器对象
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])  # 完成实际的模型配置工作
        # 不使用数据提升，所谓的提升就是从我们提供的训练数据中利用旋转、翻转、加噪声等方法创造新的
        # 训练数据，有意识的提升训练数据规模，增加模型训练量
        if not data_augmentation:
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size=batch_size,
                           nb_epoch=nb_epoch,
                           validation_data=(dataset.valid_images, dataset.valid_labels),
                           shuffle=True)
        # 使用实时数据提升
        else:
            # 定义数据生成器用于数据提升，其返回一个生成器对象datagen，datagen每被调用一
            # 次其生成一组数据（顺序生成），节省内存，其实就是python的数据生成器
            datagen = ImageDataGenerator(
                featurewise_center=False,  # 是否使输入数据去中心化（均值为0），
                samplewise_center=False,  # 是否使输入数据的每个样本均值为0
                featurewise_std_normalization=False,  # 是否数据标准化（输入数据除以数据集的标准差）
                samplewise_std_normalization=False,  # 是否将每个样本数据除以自身的标准差
                zca_whitening=False,  # 是否对输入数据施以ZCA白化
                rotation_range=20,  # 数据提升时图片随机转动的角度(范围为0～180)
                width_shift_range=0.2,  # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                height_shift_range=0.2,  # 同上，只不过这里是垂直
                horizontal_flip=True,  # 是否进行随机水平翻转
                vertical_flip=False)  # 是否进行随机垂直翻转
            # 计算整个训练样本集的数量以用于特征值归一化、ZCA白化等处理
            datagen.fit(dataset.train_images)
            # 利用生成器开始训练模型
            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                  batch_size=batch_size),
                                     samples_per_epoch=dataset.train_images.shape[0],
                                     nb_epoch=nb_epoch,
                                     validation_data=(dataset.valid_images, dataset.valid_labels))
    MODEL_PATH = './facemodel.h5'
    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path)
    def load_model(self, file_path=MODEL_PATH):
        self.model = load_model(file_path)
    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
    # 识别人脸
    def face_predict(self, image):
        # 依然是根据后端系统确定维度顺序
        if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)  # 尺寸必须与训练集一致都应该是IMAGE_SIZE x IMAGE_SIZE
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))  # 与模型训练不同，这次只是针对1张图片进行预测
        elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
            # 浮点并归一化
        image = image.astype('float32')
        image /= 255
        # 给出输入属于各个类别的概率，我们是二值类别，则该函数会给出输入图像属于0和1的概率各为多少
        result = self.model.predict_proba(image)
        print('result:', result)
        # 给出类别预测：0或者1
        result = self.model.predict_classes(image)
        # 返回类别预测结果
        return result[0]



if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)

    # 加载模型
    model = Model()
    model.load_model('./facemodel.h5')

    # 框住人脸的矩形边框颜色
    color = (0, 255, 0)
    # 捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(0)
    # 人脸识别分类器本地存储路径
    cascade_path = "C://Users//934554314//Anaconda3//Lib//site-packages//cv2//data//haarcascade_frontalface_alt2.xml"
    # 循环检测识别人脸
    while True:
        ret, frame = cap.read()  # 读取一帧视频
        if ret is True:
            # 图像灰化，降低计算复杂度
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            continue
        # 使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)
        # 利用分类器识别出哪个区域为人脸
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                # 截取脸部图像提交给模型识别这是谁
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                faceID = model.face_predict(image)
                # 如果是“我”
                faceID = 0
                if faceID == 0:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
                    # 文字提示是谁
                    cv2.putText(frame, 'liziqiang',
                                (x + 30, y + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                1,  # 字号
                                (255, 0, 255),  # 颜色
                                2)  # 字的线宽
                else:
                    pass
        cv2.imshow("识别朕", frame)
        # 等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        # 如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break
    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
