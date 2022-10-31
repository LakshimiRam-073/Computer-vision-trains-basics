from email.mime import image
import cv2 as cv
import numpy as np

img = cv.imread(r"C:\Users\Harish\Documents\MyML\Section 40 - Convolutional Neural Networks (CNN)\dataset\test_set\dogs\dog.4002.jpg")
cv.imshow('image',img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

threshold,thresh_img = cv.threshold(gray,150,255,cv.THRESH_BINARY)
cv.imshow('simple threshold',thresh_img)

adaptive = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,blockSize=11,C=9)
cv.imshow('adaptive',adaptive)

cv.waitKey(0)