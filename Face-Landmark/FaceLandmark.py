import numpy as np
import dlib
import cv2

dlib_model="C:/Users/hyebin/PycharmProjects/Face_DAR/models/dlib/shape_predictor_68_face_landmarks.dat"

# dlib.get_frontal_face_detector()로 얼굴 Detect를 불러온다
detector = dlib.get_frontal_face_detector()

# dlib.shape_predictor()은 shape_predictor_68_face_landmarks.dat 모델을 통해 얼굴에 68개의 좌표를 찍어주는 기능을 수행한다
predictor = dlib.shape_predictor(dlib_model)

img_file="C:/Users/hyebin/PycharmProjects/Face_DAR/Face_Landmark/images/Cha_Eun-Woo.jpg"
img = cv2.imread(img_file)
#img=cv2.resize(img,None,fx=3,fy=3,interpolation = cv2.INTER_LINEAR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1 -> detection 하기 전에 layer를 upscale하는데 몇번 적용할지
rects = detector(gray,1)

def draw(part):

    img_copy=img.copy()

    for rect in rects:
        points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])
        for ind, point in enumerate(points[part]):
            x = point[0,0]
            y = point[0,1]
            cv2.circle(img_copy, (x, y), 1, (255, 0, 255), -1)
            cv2.putText(img_copy, "{}".format(ind + 1), (x, y - 2),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 200, 0), 1)

    return img_copy


jawline = list(range(1, 17))
jawline_part = draw(jawline)
cv2.imshow("jawline_part",jawline_part)
# cv2.imwrite("./result/jawline_part.jpg",jawline_part)


eyebrows = list(range(17, 27))
eyebrows_part = draw(eyebrows)
cv2.imshow("eyebrows_part",eyebrows_part)
# cv2.imwrite("./result/eyebrows_part.jpg",eyebrows_part)

nose = list(range(27, 36))
nose_part = draw(nose)
cv2.imshow("nose_part",nose_part)
# cv2.imwrite("./result/nose_part.jpg",nose_part)


right_eye = list(range(36, 42))
right_eye_part = draw(right_eye)
cv2.imshow("right_eye_part",right_eye_part)
# cv2.imwrite("./result/right_eye_part.jpg",right_eye_part)


left_eye = list(range(42, 48))
left_eye_part = draw(left_eye)
cv2.imshow("left_eye_part",left_eye_part)
# cv2.imwrite("./result/left_eye_part.jpg",left_eye_part)

eyes = list(range(36, 48))
eyes_part = draw(eyes)
cv2.imshow("eyes_part",eyes_part)
# cv2.imwrite("./result/eyes_part.jpg",eyes_part)

mouth = list(range(48, 68))
mouth_part = draw(mouth)
cv2.imshow("mouth_part",mouth_part)
# cv2.imwrite("./result/mouth_part.jpg",mouth_part)

all = list(range(0, 68))
all_part = draw(all)
cv2.imshow("all_part",all_part)
# cv2.imwrite("./result/all_part.jpg",all_part)


cv2.waitKey(0)
cv2.destroyAllWindows()
