import cv2
import dlib
import numpy as np
from scipy.spatial import distance
"""
img_file = "./images/j.jpg"
img = cv2.imread(img_file)
# img = cv2.resize(img, None, fx=1.7, fy= 1.7, interpolation=cv2.CHAIN_APPROX_SIMPLE)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img_copy = img.copy()
"""

face_detector = dlib.get_frontal_face_detector()

predictor_file = "C:/Users/hyebin/PycharmProjects/Face_DAR/models/dlib/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_file)

right_eye = list(range(36, 42))
left_eye = list(range(42, 48))
eyes = list(range(36, 48))

def preprocessing_img(img):


    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    rects = face_detector(img_gray)

    for rect in rects:
        # cv2.rectangle(img_copy,(rect.left(),rect.top()),(rect.right(),rect.bottom()),(0,255,0),2)

        points = np.matrix([[point.x,point.y]for point in predictor(img_gray,rect).parts()])
        for point in points[eyes]:
            x = point[0,0]
            y = point[0,1]
            # cv2.circle(img_copy,(x,y),3,(0,255,0),-1)

        right_eye_center = np.mean(points[right_eye],axis=0).astype("int")
        right_eye_center_x = right_eye_center[0, 0]
        right_eye_center_y = right_eye_center[0, 1]

        # cv2.circle(img_copy, (right_eye_center_x,right_eye_center_y), 3, (0, 255, 0), -1)

        left_eye_center = np.mean(points[left_eye],axis=0).astype("int")
        left_eye_center_x = left_eye_center[0, 0]
        left_eye_center_y = left_eye_center[0, 1]

        # cv2.circle(img_copy, (left_eye_center_x,left_eye_center_y), 3, (0, 255, 0), -1)

        # cv2.line(img_copy, (left_eye_center_x, left_eye_center_y), (right_eye_center_x, right_eye_center_y),(0,255,0),2)
        # cv2.line(img_copy, (left_eye_center_x, left_eye_center_y), (right_eye_center_x, left_eye_center_y), (0, 255, 0), 2)
        # cv2.line(img_copy, (right_eye_center_x, right_eye_center_y), (right_eye_center_x, left_eye_center_y), (0, 255, 0), 2)

        eye_center = (int((right_eye_center_x + left_eye_center_x) // 2),
                      int(right_eye_center_y + left_eye_center_y)//2)

        # cv2.circle(img_copy,eye_center,5,(0,255,0),3)

        height = left_eye_center_y - right_eye_center_y
        bottom = left_eye_center_x - right_eye_center_x
        diagonal = distance.euclidean(left_eye_center,right_eye_center)

        degree = np.degrees(np.arctan2(height,bottom))

        scale = bottom / diagonal

        warp = cv2.getRotationMatrix2D(eye_center, degree, scale)
        warped = cv2.warpAffine(img, warp, (0,0),cv2.INTER_LINEAR )

        # cv2.imshow("warped",warped)

        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        crop_rects = face_detector(warped_gray)

        for crop_rect in crop_rects:
            crop_img = warped[crop_rect.top():crop_rect.bottom(),crop_rect.left():crop_rect.right()]

            #cv2.imshow("crop_img", crop_img)

    return crop_img

cv2.waitKey(0)
cv2.destroyAllWindows()