import cv2
import face_recognition
import numpy as np

#写真を読み込ませ、特徴量を検出する
img1 = cv2.imread("images/Messi.jpg")
rgb_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1_encoding = face_recognition.face_encodings(rgb_img1)[0]

img2 = cv2.imread("images/Ronald.jpg")
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2_encoding = face_recognition.face_encodings(rgb_img2)[0]

cv2.imshow("Img" , img1)
cv2.imshow("Img2", img2)
cv2.waitKey(0)

distance = np.linalg.norm(img1_encoding - img2_encoding)
print(f"顔の類似度:{distance:.4f}")