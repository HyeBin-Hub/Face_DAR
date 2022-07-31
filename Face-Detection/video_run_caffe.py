import cv2
from detection_caffe import detection

model_name = 'C:/Users/hyebin/PycharmProjects/Face_DAR/models/caffe/res10_300x300_ssd_iter_140000.caffemodel'
prototxt_name = 'C:/Users/hyebin/PycharmProjects/Face_DAR/models/caffe/deploy.prototxt'

min_confidence = 0.3

file_name= "C:/Users/hyebin/PycharmProjects/Face_DAR/Face_Detection/images/video.mp4"
cap = cv2.VideoCapture(file_name)

while True:
    ret, frame = cap.read()
    if not ret:
        print('x')
        break

    frame_h,frame_w=frame.shape[:2]

    detection(frame,model_name,prototxt_name,min_confidence,frame_w,frame_h)

    if cv2.waitKey(1)==27:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()