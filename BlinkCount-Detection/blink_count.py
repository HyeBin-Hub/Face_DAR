import dlib
import cv2
import numpy as np
from imutils import face_utils
from scipy.spatial import distance

predictor_file = "C:/Users/hyebin/PycharmProjects/Face_DAR/models/dlib/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_file)

detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)

codec = cv2.VideoWriter_fourcc(*"XVID")
video_fps = cap.get(cv2.CAP_PROP_FPS)
video_size = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
output_video_path = "./result/output.mp4"

video_writer = cv2.VideoWriter(output_video_path, codec, video_fps, video_size)


def eye_aspect_ratio(eye):
    p2_p6_dist = distance.euclidean(landmarks[eye][1], landmarks[eye][5])
    p3_p5_dist = distance.euclidean(landmarks[eye][2], landmarks[eye][4])
    p1_p4_dist = distance.euclidean(landmarks[eye][0], landmarks[eye][3])

    eye_ratio = (p2_p6_dist + p3_p5_dist) / (2.0 * p1_p4_dist)

    return eye_ratio


RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
EYE = list(range(36, 48))

blink_count = 0
total = 0

while 1:
    hasFrame, ImgFrmae = cap.read()
    if not hasFrame:
        print("x")
        break

    ImgFrmae_gray = cv2.cvtColor(ImgFrmae, cv2.COLOR_BGR2GRAY)

    face_detector = detector(ImgFrmae_gray)

    for face in face_detector:
        # landmarks = predictor(ImgFrmae_gray, face)
        # landmarks = face_utils.shape_to_np(landmarks)

        landmarks = np.matrix([[p.x, p.y] for p in predictor(ImgFrmae_gray, face).parts()])

        left_eye = landmarks[LEFT_EYE]
        right_eye = landmarks[RIGHT_EYE]

        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(ImgFrmae, [left_eye_hull], -1, (0, 255, 0),1)
        cv2.drawContours(ImgFrmae, [right_eye_hull], -1, (0, 255, 0), 1)

        right_eye_ratio = eye_aspect_ratio(RIGHT_EYE)
        left_eye_ratio = eye_aspect_ratio(LEFT_EYE)

        eyes = (right_eye_ratio + left_eye_ratio) / 2
        print(eyes)

        if eyes < 0.22:
            blink_count += 1
        else:
            if blink_count >= 3:
                total += 1
                print("Eye blinked")
            blink_count = 0


    text = "Blink count : {}".format(total)
    cv2.putText(ImgFrmae, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 51, 255), 2)

    video_writer.write(ImgFrmae)

    cv2.imshow("ImgFrmae", ImgFrmae)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
video_writer.release()
