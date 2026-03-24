import cv2
import numpy as np
import face_recognition
import face_recognition_models

imgMe = face_recognition.load_image_file("New folder/Test.jpg")
imgMe = cv2.cvtColor(imgMe,cv2.COLOR_BGR2RGB)
encod = face_recognition.face_encodings(imgMe)[0]
facloc=face_recognition.face_locations(imgMe)[0]

cv2.rectangle(imgMe,(facloc[3],facloc[0]),(facloc[1],facloc[2]),(255,0,122),2)

imgMe1 = face_recognition.load_image_file("New folder/Train.jpg")
imgMe1 = cv2.cvtColor(imgMe1,cv2.COLOR_RGB2BGR)
encod1 = face_recognition.face_encodings(imgMe1)[0]
facloc1=face_recognition.face_locations(imgMe1)[0]

cv2.rectangle(imgMe1,(facloc1[3],facloc1[0]),(facloc1[1],facloc1[2]),(255,0,122),2)


result = face_recognition.compare_faces([encod],encod1)
print(result)
cv2.imshow("Met",imgMe1)
cv2.imshow("Me",imgMe)
cv2.waitKey(0)
