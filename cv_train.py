#-*- coding: utf-8 -*-
"""
Created on Sun Oct  9 15:03:25 2022

@author: Harish
"""

from calendar import c
from configparser import Interpolation
import cv2 as cv 
import numpy as np

blank = np.zeros((500,500,3) ,dtype='uint8')

cv.imshow('blank' , blank)


cv.rectangle(blank ,(0,0) ,(blank.shape[1]//2,blank.shape[0]//2),color=(0,255,0),thickness=-1   )
cv.imshow('white' ,blank)

cv.circle(blank,(250,250),50,(250,0,0),thickness=-1)
cv.imshow('circle' ,blank)

cv.line(blank,(0,0) ,(250,250), color=(255,255,0) ,thickness=6)
cv.imshow('line',blank)

cv.putText(blank,"Hello my name is harissh" ,(0,250) ,cv.FONT_ITALIC,1 ,(255,255,255) ,4)
cv.imshow('My name' ,blank)

cv.waitKey(0)

'this is a game'

'this is not a game'


