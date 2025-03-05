import cv2
import sys
import time
import torch
import queue
import pygame
import threading
from typing import Optional

from __init__ import *
from spotify_audio import course_detect
from state_control import state_detect
from graphics.graphics import Graphics
from nextgenstats import run_stats


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")  # macOS Apple Silicon
    elif torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPUs
    else:
        return torch.device("cpu")


def update_frames(cap: cv2.VideoCapture, frame_queue: queue.Queue, rolling_queue: queue.Queue):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        prev_time, fps = None, 30
    else:
        frame_duration = 1 / fps
    while cap.isOpened():
        ret, new_frame = cap.read()
        if not ret:
            break
        timestamp = time.time() if fps != 0 else prev_time + frame_duration if prev_time else time.time()
        if fps == 0:
            if prev_time:
                frame_duration = timestamp - prev_time
                fps = 1 / frame_duration
            prev_time = timestamp
        max_rolling_frames = int(fps * 12)
        if not frame_queue.empty():
            frame_queue.get_nowait()
        frame_queue.put(new_frame)
        rolling_queue.put((new_frame, timestamp))
        while rolling_queue.qsize() > max_rolling_frames:
            rolling_queue.get()
    cap.release()


def main():
    device = select_device()
    gp = GPINFO()
    model_store = ModelStore(device = device)
    sp = SpotifyPlayer()
    graphics = Graphics()
    cap = cv2.VideoCapture(0)

    frame_queue = queue.Queue(maxsize=1)
    rolling_queue = queue.Queue()

    frame_thread = threading.Thread(target = update_frames, args = (cap, frame_queue, rolling_queue), daemon = True)
    course_thread = threading.Thread(target = course_detect, args=(frame_queue, model_store, sp, gp), daemon=True)
    state_thread = threading.Thread(target = state_detect, args=(frame_queue, model_store, sp, gp), daemon = True)
    stat_thread = threading.Thread(target = run_stats, args=(frame_queue, rolling_queue, model_store, sp, gp),
                                   daemon = True)


    frame_thread.start()
    course_thread.start()
    state_thread.start()
    stat_thread.start()

    try:
        graphics.run(sp, gp)  # This contains the event loop now
    except Exception as e:
        import traceback
        print("Error in graphics loop:", e)
        traceback.print_exc()  # Print the full error traceback
        pygame.quit()
        sys.exit()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
     main()

