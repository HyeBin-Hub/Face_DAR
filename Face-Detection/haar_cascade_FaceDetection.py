import cv2
import numpy as np

face_cascade_name="C:/Users/hyebin/PycharmProjects/Face_DAR/models/haarcascades/haarcascade_frontalface_alt.xml"
eyes_cascade_name="C:/Users/hyebin/PycharmProjects/Face_DAR/models/haarcascades/haarcascade_eye_tree_eyeglasses.xml"

face_cascade = cv2.CascadeClassifier(face_cascade_name)
eyes_cascade = cv2.CascadeClassifier(eyes_cascade_name)

def face_n_eyes_detect(frame,face_cascade,eyes_cascade):

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    faces = face_cascade.detectMultiScale(frame_gray)

    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
        cv2.putText(frame,'faces',(x-5,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(120,120,255),2)
        faceROI = frame_gray[y:y+h,x:x+w]

        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
            cv2.putText(frame,'and eyes',(x+60,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(51,51,255),2)
        cv2.imshow('Face detection', frame)


# img detect
img_file= "C:/Users/hyebin/PycharmProjects/Face_DAR/Face_Detection/images/img_8.png"
img=cv2.imread(img_file)
(height, width) = img.shape[:2]
ratio = 1700 / width
dimension = (1700, int(height * ratio))
img = cv2.resize(img, dimension, interpolation = cv2.INTER_AREA)

# cv2.imshow("Original Image", img)

face_n_eyes_detect(img,face_cascade,eyes_cascade)

output_file_name  = "C:/Users/hyebin/PycharmProjects/Face_DAR/Face_Detection/haar_cascade_result/haar_cascade_result_img.jpg"
cv2.imwrite(output_file_name,img)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)

# video detect
video_file = 'C:/Users/hyebin/PycharmProjects/Face_DAR/Face_Detection/images/video.mp4'

cap = cv2.VideoCapture(video_file)

codec = cv2.VideoWriter_fourcc(*'XVID')
video_fps = cap.get(cv2.CAP_PROP_FPS )
video_size=(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
video_output_path="C:/Users/hyebin/PycharmProjects/Face_DAR/Face_Detection/haar_cascade_result/haar_cascade_result_video.mp4"
video_writer=cv2.VideoWriter(video_output_path,codec,video_fps,video_size)

print("총 Frame 갯수 : ",cap.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    hasFrmae, imgFrame = cap.read()
    if hasFrmae is None:
        print('x')
        break

    face_n_eyes_detect(imgFrame,face_cascade,eyes_cascade)

    video_writer.write(imgFrame)

    if cv2.waitKey(1)==27:
        break

video_writer.release()
cap.release()

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)







