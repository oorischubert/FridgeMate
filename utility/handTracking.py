import cv2
import mediapipe as mp # pip install "mediapipe==0.10.9"
import time
import math
import numpy as np
 
 
class handDetector():
    def __init__(self, mode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, 
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
 
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
 
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
 
        return img
 
    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
 
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
 
            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
 
        return self.lmList, bbox
 
    def fingersUp(self):
        fingers = []
        # Thumb
        #print(self.lmList)
        #print(self.tipIds)
        
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
 
        # Fingers
        for id in range(1, 5):
 
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
 
        # totalFingers = fingers.count(1)
 
        return fingers
 
    def findDistance(self, p1, p2, img, draw=False,r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
 
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
 
        return length, img, [x1, y1, x2, y2, cx, cy]
    
    def findAngle(self,a,b,c):
        # calculate the lengths of the sides
        ba = math.sqrt((b[1]-a[1])**2 + (b[2]-a[2])**2)
        bc = math.sqrt((b[1]-c[1])**2 + (b[2]-c[2])**2)
        ac = math.sqrt((c[1]-a[1])**2 + (c[2]-a[2])**2)

        # check for division by zero errors
        if ba == 0 or bc == 0:
            return 0

        # calculate the angle using the law of cosines
        try:
            angle = math.acos((ba**2 + bc**2 - ac**2) / (2.0 * ba * bc))
        except ValueError:
        # handle the case where the argument to acos() is outside of the range of -1 to 1
            angle = 0
        # convert the angle to degrees
        degrees = math.degrees(angle)
        return degrees
    
    #determines if person is holding something
    def isGrasping(self) -> bool:
        """
        Determines if the hand is in a grasping position.
        Returns True if the hand is grasping, otherwise False.
        """
        if not self.lmList or len(self.lmList) < 21:
            # If no landmarks are detected or insufficient landmarks, return False
            return False

        # Check if the tips of the fingers are close to the palm
        # Compare the y-coordinates of the fingertip landmarks with their corresponding lower joints
        fingers = self.fingersUp()

        # If all fingers are down except the thumb, it indicates a grasping position
        if fingers.count(1) <= 1:  # At most one finger is up
            return True

        return False
    
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        
        success, img = cap.read()
        
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
 
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        fingers = detector.fingersUp()
 
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
 
        cv2.imshow("Image", img)
        cv2.waitKey(1)
 
 
if __name__ == "__main__":
    main()