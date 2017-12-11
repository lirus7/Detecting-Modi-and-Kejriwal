import cv2
import numpy as np
import os, os.path

face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')

for i in range(117):
    try:
        img = cv2.imread('Kejri/{}.jpg'.format(i))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x,y,w,h = face_cascade.detectMultiScale(gray)[0]
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        img = img[y:y+h, x:x+w]
        cv2.imwrite('./Kejri_faces/{}.jpg'.format(i),img)
    except:
        pass
