from email.mime import image
from statistics import median
import cv2 as cv
import numpy as np

image = cv.imread(r"C:\Users\Harish\Documents\MyML\Section 40 - Convolutional Neural Networks (CNN)\dataset\test_set\dogs\dog.4011.jpg")
cv.imshow('image',image)

average_blur = cv.blur(image,(7,7))
cv.imshow('average blur',average_blur)

gaussian = cv.GaussianBlur(image,(7,7),0)
cv.imshow('gaussian',gaussian)

media =cv.medianBlur(image,7)
cv.imshow('median blur',media)


bilateral = cv.bilateralFilter(image,10 ,35,25 )
cv.imshow('bilateral',bilateral)

cv.waitKey(0)