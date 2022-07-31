import cv2
import numpy as np

model_name = 'C:/Users/hyebin/PycharmProjects/Face_DAR/models/caffe/res10_300x300_ssd_iter_140000.caffemodel'
prototxt_name = 'C:/Users/hyebin/PycharmProjects/Face_DAR/models/caffe/deploy.prototxt'

min_confidence = 0.3

def detection(frame,model_name,prototxt_name,min_confidence,img_w,img_h):

    model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name)

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))

    model.setInput(blob)
    detections = model.forward()
    print(detections[0,0,:,:].shape)

    for detect in detections[0,0,:,:]:
        confidence = detect[2]
        if confidence > min_confidence:
            box = detect[3:7] * np.array([img_w, img_h, img_w, img_h])
            left, top, right, bottom = box.astype("int")

            text = "{:.2f}%".format(confidence * 100)
            cv2.rectangle(frame, (left, top), (right, bottom),(255, 51, 51), 2)
            cv2.putText(frame, text, (left, top-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255) , 2)

    cv2.imshow("Face Detection by dnn", frame)

# img detect
file_name = "C:/Users/hyebin/PycharmProjects/Face_DAR/Face_Detection/images/img_8.png"
img = cv2.imread(file_name)

img_h, img_w = img.shape[:2]

detection(img,model_name,prototxt_name,min_confidence,img_w,img_h)

cv2.imshow("Original Image", img)

img_output_path="C:/Users/hyebin/PycharmProjects/Face_DAR/Face_Detection/caffe_result/opencv_dnn_result_img.jpg"
cv2.imwrite(img_output_path,img)

# video detect
file_name= "C:/Users/hyebin/PycharmProjects/Face_DAR/Face_Detection/images/video.mp4"
cap = cv2.VideoCapture(file_name)

codec = cv2.VideoWriter_fourcc(*'XVID')
video_fps = cap.get(cv2.CAP_PROP_FPS )
video_size=(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
video_output_path="C:/Users/hyebin/PycharmProjects/Face_DAR/Face_Detection/caffe_result/opencv_dnn_result_video.mp4"
video_writer=cv2.VideoWriter(video_output_path,codec,video_fps,video_size)

print("총 Frame 갯수 : ",cap.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    hasFrmae, imgFrame = cap.read()
    if hasFrmae is None:
        print('x')
        break

    detection(imgFrame,model_name,prototxt_name,min_confidence,video_size[0],video_size[1])

    video_writer.write(imgFrame)

    if cv2.waitKey(1)==27:
        break

video_writer.release()
cap.release()

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)