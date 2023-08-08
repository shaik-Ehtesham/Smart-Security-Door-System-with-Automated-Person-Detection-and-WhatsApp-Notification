import cv2
import face_recognition
import numpy
import os
from datetime import datetime
import pywhatkit
import numpy as np
import winsound
path = "images"
l=[]

images = []
classname = []
mydata = []

namel = []
mylist = os.listdir(path)
for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classname.append(os.path.splitext(cl)[0])
print(classname)

def findEncodings(images1):
    encdl = []
    for img in images1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encdl.append(encode)
    return encdl

def updateDB(name):
    with open('data.csv','r+') as f:
        mydatalines = f.readlines()
        namel = []
        for line in mydatalines:
            entry = line.split(",")
            namel.append(entry[0])
        if name not in namel:
            now = datetime.now()
            ds = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{ds}')



e = findEncodings(images)
print(e)
print("encoding complete")

cap = cv2.VideoCapture(0)

while True:
    success , img = cap.read()
    imgs = cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    facecurframe = face_recognition.face_locations(imgs)
    encodecurframe = face_recognition.face_encodings(imgs, facecurframe)

    for encodef , faceloc in zip(encodecurframe, facecurframe):
        matches = face_recognition.compare_faces(e , encodef)
        print(matches)
        facedis = face_recognition.face_distance(e,encodef)
        print(facedis)
        matchIndex = np.argmin(facedis)


        if matches[matchIndex]:
            name = classname[matchIndex].upper()
            print(name)
            updateDB(name)
        else:
                now = datetime.now()
                s = now.strftime('%H:%M')
                winsound.Beep(1000,3000)

                pywhatkit.sendwhatmsg("+919381911677","UNKNOWN PERSON IS TRYING TO OPEN THE LOCKER", int(s[:2]), int(s[3:]) + 1)




    cv2.imshow("webcam", img)
    cv2.waitKey(1)
