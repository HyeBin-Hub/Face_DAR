import cv2
from detection_haar_cascade import face_n_eyes_detect

video_file = 'C:/Users/hyebin/PycharmProjects/Face_DAR/Face_Detection/images/video.mp4'

face_cascade_name="C:/Users/hyebin/PycharmProjects/Face_DAR/models/haarcascades/haarcascade_frontalface_alt.xml"
eyes_cascade_name="C:/Users/hyebin/PycharmProjects/Face_DAR/models/haarcascades/haarcascade_eye_tree_eyeglasses.xml"

face_cascade = cv2.CascadeClassifier(face_cascade_name)
eyes_cascade = cv2.CascadeClassifier(eyes_cascade_name)

cap = cv2.VideoCapture(video_file)

while True:
    hasFrmae, imgFrame = cap.read()
    if hasFrmae is None:
        print('x')
        break

    face_n_eyes_detect(imgFrame,face_cascade,eyes_cascade)

    if cv2.waitKey(1)==27:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()








