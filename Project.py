import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
path = 'New folder'
images = []
names = []
mynames = os.listdir(path)

for name in mynames:
    curr = cv2.imread(f'{path}/{name}')
    images.append(curr)
    names.append(os.path.splitext(name)[0])


def findEncoding(images):
    encodeList = []
    for img in images:

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def MarkAttendace(name):
    with open('Attendance.csv','r+') as f:
        myData = f.readlines()
        nameList = []
        for line in myData:
            entry = line.strip().split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

encodeList = findEncoding(images)

print('Encoding done')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesDetected = face_recognition.face_locations(imgS)
    encode = face_recognition.face_encodings(imgS,facesDetected)

    for encodeFace, faceLoc in zip(encode,facesDetected):
        matches = face_recognition.compare_faces(encodeList,encodeFace)
        faceDis = face_recognition.face_distance(encodeList,encodeFace)

        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = names[matchIndex].upper()
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,255),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,200),2)
            MarkAttendace(name)

    cv2.imshow("Cam",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break