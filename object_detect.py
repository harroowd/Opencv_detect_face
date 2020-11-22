import cv2
import numpy as np
import time

while True:
    vs = cv2.VideoCapture(0)
    ret, frame = vs.read()
    if ret:
        # frame = cv2.resize(frame, (700, 500))
        face_cascade = cv2.CascadeClassifier('cascade_face.xml')
        # smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'cascade_smile.xml')
        frame = cv2.flip(frame, 2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

vs.release()
cv2.destroyAllWindows()


