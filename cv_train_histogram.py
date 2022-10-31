import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


image = cv.imread(r"C:\Users\Harish\Documents\MyML\Section 40 - Convolutional Neural Networks (CNN)\dataset\test_set\dogs\dog.4003.jpg")
cv.imshow('image',image)


blank = np.zeros(image.shape[:2],dtype='uint8')
# gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
# cv.imshow('gray',gray)


mask = cv.circle(blank,(image.shape[1]//2,image.shape[0]//2),100,255,-1)
masked = cv.bitwise_and(image,image,mask=mask)
cv.imshow('mask',masked)

plt.figure()
# plt.plot(histogram)
colors = ('b','g','r')

for i,col in enumerate(colors):
    histogram = cv.calcHist([image],[i],None,[256],[0,256])
    plt.plot(histogram,color=col)
    plt.xlim([0,256])
plt.show()

cv.waitKey(0)