# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 18:14:15 2019

@author: hantq
"""
"""
from mtcnn.mtcnn import MTCNN
import cv2
img = cv2.imread("ivan.jpg")
detector = MTCNN()
print(detector.detect_faces(img)) """

def detect_faces(haar_face_cascade, colored_img, scaleFactor = 1.1):
    img_copy = np.copy(colored_img)
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    
    #let's detect multiscale (some images may be closer to camera than others) images
    faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);
    
    #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imwrite('croppingface.jpg', img_copy[y:y+h,x:x+w])
    
    
    return img_copy

import numpy as np
import cv2
from matplotlib import pyplot as plt
def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
haar_face_cascade = cv2.CascadeClassifier('E:\E\Python\caidat\pkgs\libopencv-3.4.2-h20b85fd_0\Library\etc\haarcascades\haarcascade_frontalface_alt.xml')

test1 = cv2.imread('ivan.jpg')

#convert the test image to gray image as opencv face detector expects gray images
gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)

#display the gray image using OpenCV
# cv2.imshow('Test Imag', gray_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#or if you have matplotlib installed then 
#plt.imshow(convertToRGB(test1))
#plt.imshow(gray_img, cmap='gray')
faces_detected_img = detect_faces(haar_face_cascade, test1)
plt.imshow(convertToRGB(faces_detected_img))


