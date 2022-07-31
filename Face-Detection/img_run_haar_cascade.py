import cv2
from detection_haar_cascade import face_n_eyes_detect


img_file= "C:/Users/hyebin/PycharmProjects/Face_DAR/Face_Detection/images/img_8.png"
img=cv2.imread(img_file)
(height, width) = img.shape[:2]
ratio = 1700 / width
dimension = (1700, int(height * ratio))
img = cv2.resize(img, dimension, interpolation = cv2.INTER_AREA)

cv2.imshow("Original Image", img)

face_cascade_name="C:/Users/hyebin/PycharmProjects/Face_DAR/models/haarcascades/haarcascade_frontalface_alt.xml"
eyes_cascade_name="C:/Users/hyebin/PycharmProjects/Face_DAR/models/haarcascades/haarcascade_eye_tree_eyeglasses.xml"

face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()

if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('x')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('x')
    exit(0)

face_n_eyes_detect(img,face_cascade,eyes_cascade)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)