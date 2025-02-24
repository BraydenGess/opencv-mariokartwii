import cv2
import sys
import time
from __init__ import *
from spotify_audio import run_audio

def safe_framepull():
    cap = cv2.VideoCapture(0)
    return cap

def main():
    #frame = cv2.imread('development/Images/Courses/WariosGoldMine/WariosGoldMine_57.png')
    model_store = ModelStore()
    sp = SpotifyPlayer()
    cap = safe_framepull()
    while cap.isOpened():
    #while True:
        ret, frame = cap.read()
        run_audio(frame, model_store, sp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
     main()

