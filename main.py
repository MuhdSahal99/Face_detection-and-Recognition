import csv

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#from PIL import ImageGrab
path = 'Training Images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []


    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'a+') as f:
        lnwriter = csv.writer(f)
        myDataList = f.readlines()

        nameList = [line.split(',')[0] for line in myDataList]  # Extracting names from existing lines

        if name not in nameList:
            now = datetime.now()
            current_time = now.strftime('%H:%M:%S')
            # f.writelines(f'{name},{dtString}\n')
            lnwriter.writerow([name, current_time])

        # nameList = []
        # for line in myDataList:
        #     entry = line.split(',')
        #     nameList.append(entry[0])
        #     if name not in nameList:
        #         now = datetime.now()
        #         dtString = now.strftime('%H:%M:%S')
        #         print(f'Attendance marked for {name} at {dtString}')
        #         f.writelines(f'\n{name},{dtString}')

cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera opened successfully.")


encodeListKnown = findEncodings(images)
print('Encoding Complete')


# cap = cv2.VideoCapture(0)

cv2.namedWindow('webcam', cv2.WINDOW_NORMAL)
cv2.resizeWindow('webcam', 800, 600)

while True:
    success, img = cap.read()

    if img is None:
        print("Error: Image is empty.")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches  = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            markAttendance(name)

    cv2.imshow('webcam', img)
    k = cv2.waitKey(1)
    if k == 27:  # press 'ESC' to exit
        break

cv2.destroyAllWindows()
cap.release()


