import cv2
import mediapipe as mp
import time
import hand_detectin_module as hdm
import math
from pycaw.pycaw import AudioUtilities
import numpy as np

device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
# print(f"Audio output: {device.FriendlyName}")
# print(f"- Muted: {bool(volume.GetMute())}")
#print(f"- Volume level: {volume.GetMasterVolumeLevel()} dB")
#print(f"- Volume range: {volume.GetVolumeRange()[0]} dB - {volume.GetVolumeRange()[1]} dB")

minvol=volume.GetVolumeRange()[0]
maxvol=volume.GetVolumeRange()[1]


v=volume.GetMasterVolumeLevel()   # -63.5-0 -> 400-150
volbar=np.interp(v,[-63.5,0],[400,150])
volper=np.interp(v,[-63.5,0],[0,100])

#volbar=400

cap=cv2.VideoCapture(0)
detector=hdm.handdetector()
while True:
    ret,img=cap.read()
    img=detector.findhands(img)
    lmlist1=detector.findpos(img,handno=0,draw=False)
    lmlist2=detector.findpos(img,handno=1,draw=False)
    if len(lmlist1)!=0:
        x1,y1=lmlist1[8][1],lmlist1[8][2]
        cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
    if len(lmlist2)!=0:
        x2,y2=lmlist2[8][1],lmlist2[8][2]
        cv2.circle(img,(x2,y2),15,(255,0,255),cv2.FILLED)
    if len(lmlist1)!=0 and len(lmlist2)!=0:
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
        cv2.circle(img,(int((x1+x2)/2),int((y1+y2)/2)),15,(255,0,255),cv2.FILLED)
        length=math.hypot(x2-x1,y2-y1)
        #print(length)
        vol=np.interp(length,[40,350],[minvol,maxvol])
        volbar=np.interp(length,[40,350],[400,150])
        volper=np.interp(length,[40,350],[0,100])
        #print(vol)
        #volume.SetMasterVolumeLevel(vol, None)              # uncomment while running

        if length<40:
            cv2.circle(img,(int((x1+x2)/2),int((y1+y2)/2)),15,(0,255,0),cv2.FILLED)

    cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
    cv2.rectangle(img,(50,int(volbar)),(85,400),(0,255,0),cv2.FILLED)
    cv2.putText(img,f"volume: {volper}%",(40,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)
    cv2.imshow('image',img)
    if cv2.waitKey(1) & 0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()



# volume.SetMasterVolumeLevel(-20.0, None)


# length range - 40-350
