from email.mime import image
import cv2 as cv
import numpy as np

img = cv.imread(r"C:\Users\Harish\Documents\MyML\Section 40 - Convolutional Neural Networks (CNN)\dataset\test_set\dogs\dog.4002.jpg")
cv.imshow('image',img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

lap = cv.Laplacian(gray,cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('laplacian',lap)

sobelx = cv.Sobel(gray,cv.CV_64F,1,0)
sobely= cv.Sobel(gray,cv.CV_64F,0,1)
combined = cv.bitwise_or(sobelx,sobely)
"big cock uhh..."

cv.imshow('soblx',sobelx)
cv.imshow('sobly',sobely)
cv.imshow('combined',combined)

canny = cv.Canny(gray,150,175)
cv.imshow('canny',canny)

cv.waitKey(0)


