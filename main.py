import cv2
import time
import torch
import queue
import threading
from typing import Optional

from __init__ import *
from spotify_audio import course_detect
from state_control import state_detect


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # macOS Apple Silicon
    elif torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPUs
    else:
        return torch.device("cpu")


def update_frames(cap, frame_queue):
    while cap.isOpened():
        ret, new_frame = cap.read()
        if not ret:
            break
        if not frame_queue.empty():
            frame_queue.get_nowait()
        frame_queue.put(new_frame)
    cap.release()


def main():
    device = select_device()
    model_store = ModelStore(device = device)
    sp = SpotifyPlayer()
    cap = cv2.VideoCapture(0)

    frame_queue = queue.Queue(maxsize=1)
    state_trigger = False

    frame_thread = threading.Thread(target = update_frames, args = (cap, frame_queue), daemon = True)
    course_thread = threading.Thread(target = course_detect, args=(frame_queue, model_store, sp), daemon=True)
    state_thread = threading.Thread(target = state_detect, args=(frame_queue, model_store, sp, state_trigger), daemon = True)

    frame_thread.start()
    course_thread.start()
    state_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
     main()

