'''
==============================
test3:找出图片中的条形码或二维码
（ps.仅识别条形码成功）
==============================
'''
import numpy as np
import argparse
import cv2

# print("请输入解码图片完整名称：")
#code_name = input('>>:').strip()
print("正在识别：")
#image = cv2.imread("test2.jpg")
image = cv2.imread('C:\\Users\\934554314\\Desktop\\ss.JPG')
cv2.imshow("ScanQRcodeTest", image)
cv2.waitKey(0)
# 灰度
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("ScanQRcodeTest", gray)
cv2.waitKey(0)

# 使用opencv自带的Sobel算子进行过滤
gradX = cv2.Sobel(gray, ddepth=cv2.cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=cv2.cv2.CV_32F, dx=0, dy=1, ksize=-1)

# 将过滤得到的X方向像素值减去Y方向的像素值
gradient = cv2.subtract(gradX, gradY)
# 先缩放元素再取绝对值，最后转换格式为8bit型
gradient = cv2.convertScaleAbs(gradient)
# 均值滤波取二值化
blurred = cv2.blur(gradient, (17, 17))
(_, thresh) = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

cv2.imshow("thresh", thresh)
cv2.waitKey(0)
# 腐蚀和膨胀的函数
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4)
cv2.imshow("closed", closed)
cv2.waitKey(0)
# 找到边界findContours函数
cnts,hierarchy = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 计算出包围目标的最小矩形区域
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))

#======显示=======
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
cv2.imshow("ScanQRcodeTest111", image)
cv2.waitKey(0)
