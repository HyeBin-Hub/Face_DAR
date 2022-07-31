import cv2
import numpy as np
import dlib


dlib_model="C:/Users/hyebin/PycharmProjects/Face_DAR/models/dlib/shape_predictor_68_face_landmarks.dat"

predictor=dlib.shape_predictor(dlib_model)
detector=dlib.get_frontal_face_detector()

# ----------------------------------------------------------------------------
# Face detection
img_file="C:/Users/hyebin/PycharmProjects/Face_DAR/Face_Landmark/images/Cha_Eun-Woo.jpg"
img=cv2.imread(img_file)
# img=cv2.resize(img,None,fx=1.8,fy=1.8,interpolation=cv2.INTER_LINEAR)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

rects=detector(img_gray)

img_copy=img.copy()

left=rects[0].left()
top=rects[0].top()
right=rects[0].right()
bottom=rects[0].bottom()

cv2.rectangle(img_copy,(left,top),(right,bottom),(0,255,0),3)

cv2.imshow("ori_img",img_copy)
# ----------------------------------------------------------------------------
# Face Landmark
img_copy_1=img.copy()

for ind,i in enumerate(predictor(img_gray,rects[0]).parts(),1):
    cv2.circle(img_copy_1,(i.x,i.y),3,(0,225,0),-1)
    cv2.putText(img_copy_1,str(ind),(i.x,i.y-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

cv2.imshow("FaceLandmark",img_copy_1)

img_copy_2=img.copy()

left_eye=list(range(36,42))
right_eye=list(range(42,48))
eyes=list(range(36,48))

points=np.matrix([ [i.x,i.y ] for i in predictor(img_gray,rects[0]).parts()])
# print(points[0,0])
# pointss=[ [i.x,i.y ] for i in predictor(img_gray,rects[0]).parts()]
# print(pointss[0][0])

for i in eyes:
    part = points[i]

    cv2.circle(img_copy_2, (part[0,0], part[0,1]), 3, (0, 225, 0), -1)
    cv2.putText(img_copy_2, str(i+1), (part[0,0], part[0,1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

cv2.imshow("EyesPoints",img_copy_2)
# ----------------------------------------------------------------------------
# Face Alignment

img_copy_3 = img.copy()

left_eye_center = np.mean(points[left_eye],axis=0).astype("int")
cv2.circle(img_copy_3, (left_eye_center[0,0], left_eye_center[0,1]), 5, (255, 255, 51), -1)
right_eye_center = np.mean(points[right_eye],axis=0).astype("int")
cv2.circle(img_copy_3, (right_eye_center[0,0], right_eye_center[0,1]), 5, (255, 255, 51), -1)

cv2.line(img_copy_3,(left_eye_center[0,0],left_eye_center[0,1]),(right_eye_center[0,0],right_eye_center[0,1]),(0,225,0),2)


cv2.circle(img_copy_3,(left_eye_center[0,0],right_eye_center[0,1]),5,(255, 255, 51), -1)
cv2.line(img_copy_3,(left_eye_center[0,0],left_eye_center[0,1]),(left_eye_center[0,0],right_eye_center[0,1]),(0,225,0),2)
cv2.line(img_copy_3,(right_eye_center[0,0],right_eye_center[0,1]),(left_eye_center[0,0],right_eye_center[0,1]),(0,225,0),2)

height=right_eye_center[0,1]-left_eye_center[0,1]
cv2.putText(img_copy_3,str(height),(left_eye_center[0,0]-30,right_eye_center[0,1]-13),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
bottom=right_eye_center[0,0]-left_eye_center[0,0]
cv2.putText(img_copy_3,str(bottom),(left_eye_center[0,0]+80,right_eye_center[0,1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)

degrees=np.degrees(np.arctan2(right_eye_center[0,1]-left_eye_center[0,1],right_eye_center[0,0]-left_eye_center[0,0]))
cv2.putText(img_copy_3,str(degrees),(right_eye_center[0,0]+10, right_eye_center[0,1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)

img_copy_4=img.copy()

eyes_center=(int((left_eye_center[0,0]+right_eye_center[0,0])//2),
            int((left_eye_center[0,1]+right_eye_center[0,1])//2))
cv2.circle(img_copy_3,eyes_center,5,(255, 255, 51), -1)

diagonal=np.sqrt((left_eye_center[0,0]-right_eye_center[0,0])**2+(left_eye_center[0,1]-right_eye_center[0,1])**2)
cv2.putText(img_copy_3,str(diagonal),(eyes_center[0]-20,eyes_center[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)

cv2.imshow("DrawImg",img_copy_3)

sacle=bottom/diagonal
warp=cv2.getRotationMatrix2D(eyes_center,degrees,sacle )
wraped=cv2.warpAffine(img_copy_4,warp,(0,0),cv2.INTER_LINEAR  )

cv2.imshow("WarpedImg",wraped)

wraped_gray=cv2.cvtColor(wraped,cv2.COLOR_BGR2GRAY)
re_rects=detector(wraped_gray)
crop_img = wraped[re_rects[0].top():re_rects[0].bottom(),re_rects[0].left():re_rects[0].right()]

cv2.imshow("CropImg",crop_img)

cv2.waitKey(0)
cv2.destroyAllWindows()