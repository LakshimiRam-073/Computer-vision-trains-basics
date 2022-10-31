from tokenize import blank_re
import numpy as np
import cv2 as cv


image = cv.imread(r"C:\Users\Harish\Documents\MyML\Section 40 - Convolutional Neural Networks (CNN)\dataset\test_set\dogs\dog.4002.jpg")
cv.imshow('image',image)


blank = np.zeros(image.shape ,dtype='uint8')
gray =cv.cvtColor(image,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

blur =cv.GaussianBlur(gray,(3,3) ,cv.BORDER_DEFAULT)
cv.imshow('blur',blur)

canny = cv.Canny(blur, 125 ,175)
cv.imshow('canny',canny)

ret ,threshold = cv.threshold(gray,175,255,cv.THRESH_BINARY)
cv.imshow('thres',threshold)
countours, hierarchy = cv.findContours(canny, cv.RETR_LIST ,cv.CHAIN_APPROX_SIMPLE)

cv.drawContours(blank ,countours,-1 ,(255,0,0),1)
cv.imshow('counters',blank)
print(len(countours))
cv.waitKey(0)