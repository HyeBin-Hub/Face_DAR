# EAR(Eye Aspect ratio)알고리즘을 이용한 운전자 졸음 운전 방지 시스템

# 눈을 감게 되면 눈의 세로 비율이 작아지고, EAR 비율 역시 작아진다
# 사람이 깜박이면 눈의 종횡비가 급격히 감소하여 0에 가까워진다
# 사람이 졸리면 눈을 감거나 눈을 조그마하게 뜨는 행동을 하게 되므로 EAR값을 이용하여
# 주기적으로 낮아지면 알람을 울리게 하는 졸음 감지 시스템을 구현한다

# https://ultrakid.tistory.com/12

import cv2
import numpy as np
import dlib
import pygame
from scipy.spatial import distance
from imutils import face_utils

predictor_model = "C:/Users/hyebin/PycharmProjects/Face_DAR/models/dlib/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_model)

detector = dlib.get_frontal_face_detector()

RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
EYE = list(range(36, 48))
"""
img_file="./images/me.png"
img=cv2.imread(img_file)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

face_detector = detector(img_gray)

for face in face_detector:
    landmarks = predictor(img_gray,face)
    landmarks = face_utils.shape_to_np(landmarks)

    p2_p6_dist = distance.euclidean(landmarks[LEFT_EYE][1],landmarks[LEFT_EYE][5])
    p3_p5_dist = distance.euclidean(landmarks[LEFT_EYE][2],landmarks[LEFT_EYE][4])
    p1_p4_dist = distance.euclidean(landmarks[LEFT_EYE][0],landmarks[LEFT_EYE][3])

    eye_ratio = (p2_p6_dist + p3_p5_dist) / (2.0 * p1_p4_dist)
"""

pygame.mixer.init()
pygame.mixer.music.load('./sound/Warning_Long.mp3')

cap = cv2.VideoCapture(0)

codec = cv2.VideoWriter_fourcc(*"XVID")
video_fps = cap.get(cv2.CAP_PROP_FPS)
video_size = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
output_video_path = "./result/output.mp4"

video_writer = cv2.VideoWriter(output_video_path, codec, video_fps, video_size)

RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
EYE = list(range(36, 48))


def eye_aspect_ratio(eye):
    p2_p6_dist = distance.euclidean(landmarks[eye][1], landmarks[eye][5])
    p3_p5_dist = distance.euclidean(landmarks[eye][2], landmarks[eye][4])
    p1_p4_dist = distance.euclidean(landmarks[eye][0], landmarks[eye][3])

    eye_ratio = (p2_p6_dist + p3_p5_dist) / (2.0 * p1_p4_dist)

    return eye_ratio


cnt = 0
color = (255, 0, 0)
while 1:
    hasFrame, imgFrame = cap.read()
    if not hasFrame:
        print("x")
        break

    imgFrame_gray = cv2.cvtColor(imgFrame, cv2.COLOR_BGR2GRAY)

    face_detector = detector(imgFrame_gray)

    text = ""

    for face in face_detector:
        landmarks = predictor(imgFrame_gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_eye = eye_aspect_ratio(LEFT_EYE)
        right_eye = eye_aspect_ratio(RIGHT_EYE)

        eye = (left_eye + right_eye) / 2

        if eye < 0.25:
            imgFrame = cv2.cvtColor(imgFrame, cv2.COLOR_BGR2GRAY)
            text = "Warning!"
            cnt += 1
            color = (0, 0, 255)
        else:
            cnt = 0
            color = (255, 0, 0)

        if cnt > 7:
            if (pygame.mixer.music.get_busy() == False):
                pygame.mixer.music.play()

    cv2.putText(imgFrame, text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
    video_writer.write(imgFrame)

    cv2.imshow("imgFrame", imgFrame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
video_writer.release()



