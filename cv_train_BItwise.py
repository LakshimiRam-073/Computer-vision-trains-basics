
import numpy as np
import cv2 as cv

blank = np.zeros((400,400),dtype='uint8')

rectangl = cv.rectangle(blank.copy(),(30,30),(370,370),255,-1)
circl = cv.circle(blank.copy(),(200,200),200,255,-1)

cv.imshow('circle',circl)
cv.imshow('rectangle',rectangl)

AND = cv.bitwise_and(circl,rectangl)
cv.imshow('and',AND)

OR = cv.bitwise_or(circl,rectangl)
cv.imshow('OR',OR)

XOR = cv.bitwise_xor(circl,rectangl)
cv.imshow('XOR',XOR)


not_rectangel = cv.bitwise_not(rectangl)
cv.imshow('rectangle -> not',not_rectangel)


not_circle = cv.bitwise_not(circl)
cv.imshow('circle -> not',not_circle)
cv.waitKey(0)