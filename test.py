import os
import cv2
import time
import queue
import threading
import random
from development.countdown_detection.countdown_detector import CountdownDetector
from development.character_detection.character_detector import CharacterDetector
from development.vehicle_detection.vehicle_detector import VehicleDetector

import os
import multiprocessing

#frame = cv2.imread('development/Images/Characters/Wario+Waluigi+DonkeyKong+Bowser_0.png')
#frame2 = cv2.imread('development/Images/RawImages/2characterWario.png')
#frame2 = cv2.imread('development/Images/RawImages/DelfinoSquareF_17.png')
#frame3 = cv2.imread('development/Images/RawImages/2characters.png')
#model = CountdownDetector(model_path = 'production/models/countdown_detection.pth')
#(445, 485, 120, 420)
#model = CharacterDetector(model_path = 'production/models/character_classifier.pth')
#print(model.predict(frame = frame, players = 4))
#print(model.predict(frame = frame2, players = 2))
#print(model.predict(frame = frame3, players = 2))
#frame = frame[445:485, 120:420]
#frame_2 = cv2.resize(frame2[430:485, 110:570],(300,40))
#frame_3 = cv2.resize(frame2[795:850, 110:570],(300,40))

#model = VehicleDetector(model_path = 'production/models/vehicle_classifier.pth')
#print(model.predict(frame = frame, players = 4))
#print(model.predict(frame = frame2, players = 2))


#frame3 = frame3[730:880, 100:850]

#Characters
#frame = frame[445:485, 120:420]
#frame_2 = cv2.resize(frame2[785:855, 110:570],(300,40))
#frame_3 = cv2.resize(frame2[420:490, 110:570],(300,40))
#Vehicles432, 512, 305, 855
#frame = frame[432:512, 305:855]
#frame_2 = cv2.resize(frame2[420:495, 425:920],(550,80))
#frame_3 = cv2.resize(frame2[810:885, 425:920],(550,80))

frame = cv2.imread('development/Images/Countdown/GO/GO_DKSummit_56.png')
frame2 = cv2.imread('development/Images/RawImages/GO_MOOMOO.png')

frame_1 = frame[190:340, 100:850]
frame_2 = frame2[190:340, 570:1320]

model = CountdownDetector(model_path = 'production/models/countdown_detection.pth')
print(model.predict(frame, players = 4))
print(model.predict(frame2, players = 2))

cv2.imshow('Sliced Frame', frame_1)
cv2.imshow('Sliced Frame2', frame_2)
#cv2.imshow('Sliced Frame3', frame_3)


cv2.waitKey(0)
cv2.destroyAllWindows()
