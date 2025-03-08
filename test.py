import os
import cv2
import time
import queue
import threading
import random
from development.countdown_detection.countdown_detector import CountdownDetector
import os
import multiprocessing



frame = cv2.imread('production/referenceimages/player_counts/4player.png')
frame = frame[55:115,535:625]
cv2.imshow('Sliced Frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()


frame2 = cv2.imread('development/Images/RawImages/DelfinoSquareF_17.png')
frame3 = cv2.imread('development/Images/RawImages/KoopaCapeF_12.png')
model = CountdownDetector(model_path = 'production/models/countdown_detection.pth')

frame = frame[:,12:len(frame[1])-24]
#frame = frame[190:340, 100:850]
#frame2 = frame2[190:340, 1035:1785]
#frame3 = frame3[730:880, 100:850]



cv2.imshow('Sliced Frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()
cv2.imshow('Sliced Frame2', frame2)
cv2.imshow('Sliced Frame3', frame3)


# Wait for a key press to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()