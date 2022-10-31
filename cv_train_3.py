import imghdr
from turtle import width
import cv2 as cv
import numpy as np


image = cv.imread(r"C:\Users\Harish\Documents\MyML\Section 40 - Convolutional Neural Networks (CNN)\dataset\test_set\dogs\dog.4002.jpg")
cv.imshow('image',image)

def translated(img , x , y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img,transMat,dimensions)

translated_img = translated(image,100 ,-100)
cv.imshow('translated',translated_img)
# +x---> right    -x---->left
# +y ---> down     -y--->up

def rotated(img, angle , rotated_point =None):
    (hieght,width) =img.shape[:2]

    if rotated_point== None:
        rotated_point =(width//2, hieght//2)

    rotmat = cv.getRotationMatrix2D(rotated_point,angle,1.0)
    dimension =(width,hieght)
    return cv.warpAffine(img,rotmat,dimension)


rotated_img = rotated(image,-45)
cv.imshow('rotated',rotated_img)


resize = cv.resize(image,(300,400),interpolation=cv.INTER_AREA)
cv.imshow('resized' ,resize)


flip = cv.flip(image,-1)
cv.imshow('flip', flip)




cv.waitKey(0)