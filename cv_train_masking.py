from turtle import circle
import numpy as np
import cv2 as cv

image = cv.imread(r"C:\Users\Harish\Documents\MyML\Section 40 - Convolutional Neural Networks (CNN)\dataset\test_set\dogs\dog.4002.jpg")
cv.imshow('image',image)

blank = np.zeros(image.shape[:2],dtype='uint8')

mask = cv.circle(blank,(image.shape[1]//2,image.shape[0]//2 -100),100,255,-1)
cv.imshow('mask',mask)

masked = cv.bitwise_and(image,image,mask=mask)
cv.imshow('masked image ',masked)
cv.waitKey(0)