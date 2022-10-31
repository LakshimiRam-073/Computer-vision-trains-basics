
import numpy as np
import cv2 as cv
# import matplotlib.pyplot as plt

image = cv.imread(r"C:\Users\Harish\Documents\MyML\Section 40 - Convolutional Neural Networks (CNN)\dataset\test_set\dogs\dog.4003.jpg")
cv.imshow('image',image)

gray =cv.cvtColor(image , cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)
cv.imshow('hsv',hsv)

lab = cv.cvtColor(image,cv.COLOR_BGR2LAB)
cv.imshow('lab',lab)

rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
cv.imshow('rgb',rgb)


# plt.imshow(image)
# plt.show()

'nothing to be worried'


cv.waitKey(0)