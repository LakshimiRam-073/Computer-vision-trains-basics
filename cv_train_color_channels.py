import cv2 as cv
import numpy as np


image = cv.imread(r"C:\Users\Harish\Documents\MyML\Section 40 - Convolutional Neural Networks (CNN)\dataset\test_set\cats\cat.4008.jpg")
cv.imshow('image',image)

blank = np.zeros(image.shape[:2],dtype='uint8')
b,g,r = cv.split(image)

blue = cv.merge([b,blank,blank])
green = cv.merge([blank,g,blank])
red =cv.merge([blank,blank,r])

cv.imshow('blue',blue)
cv.imshow('green',green)
cv.imshow('red',red)
cv.waitKey(0)