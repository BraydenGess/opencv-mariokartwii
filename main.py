import cv2
import sys
import time
from __init__ import *
from spotify_audio import run_audio

def safe_framepull():
    cap = cv2.VideoCapture(0)
    return cap

def main():
    model_store = ModelStore()
    sp = SpotifyPlayer()
    cap = safe_framepull()
    while cap.isOpened():
        ret, frame = cap.read()
        run_audio(frame, model_store, sp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
     main()

