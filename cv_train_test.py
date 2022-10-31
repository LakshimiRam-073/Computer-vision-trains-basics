from cProfile import label
from cgi import test
from email.mime import image
import cv2 as cv
import numpy as np
import os
DIR =r'C:\Users\Harish\Documents\face recognition\data\train'
people =[]
for person in os.listdir(DIR):
    people.append(person)


face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_recog.yml')

test_image = cv.imread(r"C:\Users\Harish\Documents\face recognition\data\val\mindy_kaling\httpcdnpastemagazinecomwwwarticlesmindykalingndbookjpg.jpg")
gray = cv.cvtColor(test_image,cv.COLOR_BGR2GRAY)
haar = cv.CascadeClassifier('haar.xml')

face_rect = haar.detectMultiScale(gray,1.1,4)

for (x,y,w,h) in face_rect:
    face_roi = gray[y:y+h,w:w+x]

    label,confidence = face_recognizer.predict(face_roi)
    print(f'Label of {people[label]},with a confidence of {confidence}')


    cv.putText(test_image,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(255,0,0),2)
    cv.rectangle(test_image,(x,y),(x+w,y+h),(255,0,0),2)


cv.imshow('Detected face',test_image)

cv.waitKey(0)


