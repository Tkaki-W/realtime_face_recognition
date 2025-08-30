import cv2
import face_recognition

#写真を読み込ませ、特徴量を検出する
img = cv2.imread("../images/messi.webp")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

cv2.imshow("Img" , img)
cv2.waitKey(0)