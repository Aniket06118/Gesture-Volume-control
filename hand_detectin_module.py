import cv2
import mediapipe as mp
import time

class handdetector():
    def __init__(self,mode=False,maxhands=2,detectcon=0.5,trackcon=0.5):
        self.mode=mode
        self.maxhands=maxhands
        self.detectcon=detectcon
        self.trackcon=trackcon

        self.mphands=mp.solutions.hands
        self.hands=self.mphands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxhands,
                                        min_detection_confidence=self.detectcon,
                                        min_tracking_confidence=self.trackcon)
        self.mpdraw=mp.solutions.drawing_utils

    def findhands(self,img,draw=True):
        rgbimg=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(rgbimg)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img,handlms,self.mphands.HAND_CONNECTIONS)
               
        return img
    
    def findpos(self,img,handno=0,draw=True):
        lmlist=[]
        if self.results.multi_hand_landmarks:
            if handno < len(self.results.multi_hand_landmarks):
                myhand=self.results.multi_hand_landmarks[handno]
                for id,lm in enumerate(myhand.landmark):
                                    h,w,c=img.shape
                                    cx,cy=int(lm.x*w),int(lm.y*h)
                                    lmlist.append([id,cx,cy])
                                    if draw:
                                        cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)
    
        return lmlist



def main():
    cap=cv2.VideoCapture(0)
    ctime=0
    ptime=0
    detector=handdetector()

    while True:
        ret,img=cap.read()
        img=detector.findhands(img)
        lmlist=detector.findpos(img)
        if len(lmlist) !=0:
             print(lmlist[4])
        #print(lmlist)
        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime
        cv2.putText(img,f'fps:{fps}',(10,70),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
        cv2.imshow("camera feed",img)
        if cv2.waitKey(1) & 0xFF==27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
    