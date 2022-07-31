import cv2
import numpy as np

def face_n_eyes_detect(frame,face_cascade,eyes_cascade):

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    faces = face_cascade.detectMultiScale(frame_gray)

    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
        cv2.putText(frame,'faces',(x-5,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(120,120,255),2)
        faceROI = frame_gray[y:y+h,x:x+w]

        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
            cv2.putText(frame,'eyes',(x+60,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(51,51,255),2)
        cv2.imshow('Face detection', frame)