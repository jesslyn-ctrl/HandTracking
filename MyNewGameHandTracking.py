import cv2
import mediapipe as mp
import time
import HandTrackingModule as hdm

cap = cv2.VideoCapture(0)
prevTime = 0
detector = hdm.handDetector()
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList[4])

    curTime = time.time()
    fps = 1 / (curTime - prevTime)
    prevTime = curTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)