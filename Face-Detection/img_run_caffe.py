import cv2
import numpy as np
from detection_caffe import detection

model_name = 'C:/Users/hyebin/PycharmProjects/Face_DAR/models/caffe/res10_300x300_ssd_iter_140000.caffemodel'
prototxt_name = 'C:/Users/hyebin/PycharmProjects/Face_DAR/models/caffe/deploy.prototxt'

min_confidence = 0.3

file_name = "C:/Users/hyebin/PycharmProjects/Face_DAR/Face_Detection/images/img_8.png"
img = cv2.imread(file_name)

img_h, img_w = img.shape[:2]

cv2.imshow("Original Image", img)

detection(img,model_name,prototxt_name,min_confidence,img_w,img_h)

cv2.waitKey(0)
cv2.destroyAllWindows()