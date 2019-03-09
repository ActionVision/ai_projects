import cv2


def detect():
    face_cascade = cv2.CascadeClassifier("C://Users//934554314//Anaconda3//Lib//site-packages//cv2//data//haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("C://Users//934554314//Anaconda3//Lib//site-packages//cv2//data//haarcascade_eye.xml")
    camera = cv2.VideoCapture(0)
    while True and cv2.waitKey(1) == -1:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, 0, (40, 40))
            for (ex, ey, ew, eh) in eyes:
                # 由于是在原图片上修改,所以需要加上原来坐标
                cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
        cv2.imshow("camera", frame)
    camera.release()
    cv2.destroyAllWindows()


detect()
