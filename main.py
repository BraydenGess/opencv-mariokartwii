import cv2
from __init__ import *
from typing import Optional
from spotify_audio import run_audio

def safe_framepull() -> cv2.VideoCapture:
    try:
        cap = cv2.VideoCapture(0)
        return cap
    except Exception as e:
        return e


def safe_opencheck(cap: cv2.VideoCapture) -> bool:
    try:
        return cap.isOpened()
    except Exception as e:
        return False


def main():
    model_store = ModelStore()
    sp = SpotifyPlayer()
    cap = safe_framepull()

    while safe_opencheck(cap):
        ret, frame = cap.read()
        run_audio(frame = frame, model_store = model_store, sp = sp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
     main()

