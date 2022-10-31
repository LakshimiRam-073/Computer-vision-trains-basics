from calendar import c
import cv2 as cv
import numpy as np

img = cv.imread(r"C:\Users\Harish\Documents\MyML\Section 40 - Convolutional Neural Networks (CNN)\dataset\test_set\dogs\dog.4002.jpg")
cv.imshow('dog' ,img)

grey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',grey)


blur = cv.GaussianBlur(img , (3,3) ,cv.BORDER_DEFAULT)
cv.imshow('blur' , blur)

cascade = cv.Canny(blur, 125, 175)
cv.imshow('canny' ,cascade)


dilating = cv.dilate(cascade,(3,3) ,iterations=3)
cv.imshow('dilated' , dilating)


erodeing = cv.erode(dilating , (3,3), iterations=3)
cv.imshow('erode',erodeing)


croped = img[200:300 ,300:400]
cv.imshow('cropped',croped)
cv.waitKey(0)