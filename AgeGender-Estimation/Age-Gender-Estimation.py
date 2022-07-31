import cv2
import numpy as np
import pandas as pd
from preprocessing import preprocessing_img


face_caffe_model = "C:/Users/hyebin/PycharmProjects/Face_DAR/models/caffe/res10_300x300_ssd_iter_140000.caffemodel"
face_prototxt = "C:/Users/hyebin/PycharmProjects/Face_DAR/models/caffe/deploy.prototxt"

age_caffe_model = "C:/Users/hyebin/PycharmProjects/Face_DAR/models/caffe/age_net.caffemodel"
age_prototxt = "C:/Users/hyebin/PycharmProjects/Face_DAR/models/caffe/age_deploy.prototxt"

gender_caffe_model = "C:/Users/hyebin/PycharmProjects/Face_DAR/models/caffe/gender_net.caffemodel"
gender_prototxt = "C:/Users/hyebin/PycharmProjects/Face_DAR/models/caffe/gender_deploy.prototxt"

face_net = cv2.dnn.readNetFromCaffe(face_prototxt,face_caffe_model)
age_net = cv2.dnn.readNetFromCaffe(age_prototxt,age_caffe_model)
gender_net = cv2.dnn.readNetFromCaffe(gender_prototxt,gender_caffe_model)

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male','Female']

img_file = "./images/jin.jpeg"
img = cv2.imread(img_file)
r= 500 / img.shape[0]
# img = cv2.resize(img, (int(img.shape[1]*r),500))
img = cv2.resize(img, None, fx=0.7, fy=0.7, interpolation=cv2.CHAIN_APPROX_SIMPLE)
img_h, img_w = img.shape[:2]
print(img_h, img_w)

# cv2.imshow("imgsss",img)

img_copy = img.copy()

draw_img = img.copy()


blob = cv2.dnn.blobFromImage(img,1.0, (300,300),swapRB=False,crop=False)
face_net.setInput(blob)
face_result = face_net.forward()
# print(face_result.shape)

for face_detect in face_result[0,0,:,:]:
    score = face_detect[2]
    if score > 0.5:
        box = face_detect[3:7] * np.array([img_w, img_h, img_w, img_h])
        left, top, right, bottom = box.astype("int")
        cv2.rectangle(img_copy,(left,top),(right,bottom),(0,255,0),2)

        #face_img = img_copy[top:bottom,left:right]

        face_img = preprocessing_img(draw_img)

        print(face_img)

        cv2.imshow("face_img", face_img)


        face_blob = cv2.dnn.blobFromImage(face_img,1.0, (227, 227),swapRB=False,crop=False)

        age_net.setInput(face_blob)
        age_result = age_net.forward()
        age_index = age_result[0].argmax()
        age = age_list[age_index]

        gender_net.setInput(face_blob)
        gender_result = gender_net.forward()
        gender_index = gender_result[0].argmax()
        gender = gender_list[gender_index]

        text_age = "Age : {} ".format(age)
        cv2.putText(img_copy,text_age ,(left-50 ,top -10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

        text_gender = "Gender : {}".format(gender)
        cv2.putText(img_copy,text_gender ,(left-50 ,top -35),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

cv2.imshow("img",img_copy)

img_name = img_file[img_file.rfind("/")+1:]

output_dir = "./result/"+img_name

cv2.imwrite(output_dir,img_copy)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)


