import numpy as np
import cv2
import dlib
import math

left_eye=list(range(36,42))
right_eye=list(range(42,48))
eyes=list(range(36,48))

dlib_model="C:/Users/hyebin/PycharmProjects/Face_DAR/models/dlib/shape_predictor_68_face_landmarks.dat"
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(dlib_model)

img_file="C:/Users/hyebin/PycharmProjects/Face_DAR/Face_Landmark/images/Cha_Eun-Woo.jpg"
img=cv2.imread(img_file)
# img=cv2.resize(img,None,fx=2.5, fy=2.5,interpolation = cv2.INTER_LINEAR)
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img_h,img_w=img.shape[:2]

rects=detector(gray_img,1)

img_copy=img.copy()

for ind,rect in enumerate(rects):

    cv2.rectangle(img_copy, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)

    points=np.matrix([[p.x,p.y] for p in predictor(gray_img,rect).parts()])

    for point in points[eyes]:
        x=point[0,0]
        y=point[0,1]
        cv2.circle(img_copy,(x,y),3,(51,255,255),-1)

    left_eye_center=np.mean(points[left_eye],axis=0).astype("int")
    cv2.circle(img_copy,(left_eye_center[0,0],left_eye_center[0,1]),5,(0,0,255),-1)

    right_eye_center=np.mean(points[right_eye],axis=0).astype("int")
    cv2.circle(img_copy, (right_eye_center[0, 0], right_eye_center[0, 1]), 5, (0, 0, 255), -1)

    cv2.line(img_copy,(left_eye_center[0,0],left_eye_center[0,1]),
             (right_eye_center[0, 0], right_eye_center[0, 1]),(153,0,153),4)

    cv2.circle(img_copy,(right_eye_center[0, 0],left_eye_center[0,1]),5,(255, 0, 255), -1)

    cv2.line(img_copy, (left_eye_center[0,0],left_eye_center[0,1]),
             (right_eye_center[0, 0],left_eye_center[0,1]), (255, 204, 255), 2)
    cv2.line(img_copy, (right_eye_center[0, 0], right_eye_center[0, 1]),
             (right_eye_center[0, 0],left_eye_center[0,1]), (255, 204, 255), 2)

    eye_delta_x = right_eye_center[0,0] - left_eye_center[0,0]
    eye_delta_y = right_eye_center[0,1] - left_eye_center[0,1]
    degree = np.degrees(np.arctan2(eye_delta_y,eye_delta_x)) - 180

    eye_distance = np.sqrt((eye_delta_x ** 2) + (eye_delta_y ** 2))
    aligned_eye_distance = left_eye_center[0,0] - right_eye_center[0,0]
    scale = aligned_eye_distance / eye_distance

    eyes_center = (int((left_eye_center[0,0] + right_eye_center[0,0]) // 2),
           int( (left_eye_center[0,1] + right_eye_center[0,1]) // 2))
    cv2.circle(img_copy, eyes_center, 5, (255, 0, 0), -1)

    M = cv2.getRotationMatrix2D(eyes_center, degree, scale)
    text="{:.5f}".format(degree)
    cv2.putText(img_copy,text , (int(left_eye_center[0, 0])+30, int(left_eye_center[0, 1])-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    image_origin=img.copy()

    warped = cv2.warpAffine(image_origin,M, (img_w, img_h),flags=cv2.INTER_CUBIC)
    cv2.imshow("warped_img", warped)
    cv2.imwrite("./result/warped_img.jpg", warped)

    crop_img = warped[rect.top():rect.bottom(), rect.left():rect.right()]
    cv2.imshow("crop_img", crop_img)
    cv2.imwrite("./result/crop_img.jpg",crop_img)

cv2.imshow("img_copy", img_copy)
cv2.imwrite("./result/img.jpg",img_copy)

cv2.waitKey(0)
cv2.destroyAllWindows()
