import cv2
import sys
import torch
import queue
import pygame
import threading
import collections
from collections import deque

from __init__ import *
from spotify_audio import course_detect
from state_control import state_detect
from nextgenstats import run_stats
from graphics.graphics import Graphics


def select_device() -> torch.device:
    """
    Selects the best available computing device for PyTorch

    Returns: torch.device
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")   # macOS Apple Silicon
    elif torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPUs
    return torch.device("cpu")


def update_frames(cap: cv2.VideoCapture, frame_queue: queue.Queue, rolling_queue: collections.deque):
    """"
    Continuously reads frames from a video capture source and updates the frame queues

    Parameters:
    - cap (cv2.VideoCapture): The video capture source
    - frame_queue (queue.Queue): A queue holding the latest captured frame
    - rolling_queue (queue.Queue): A queue storing frames with timestamps for historical reference
    """
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    buffer_seconds = 24
    max_rolling_frames = int(fps * buffer_seconds)
    frame_count = 0

    while cap.isOpened():
        ret, new_frame = cap.read()
        if not ret:
            break

        # Keep only the latest frame in frame_queue
        if not frame_queue.empty():
            frame_queue.get_nowait()
        frame_queue.put(new_frame)

        # Maintain rolling queue buffer
        rolling_queue.append((new_frame, frame_count))
        if len(rolling_queue) > max_rolling_frames:
            rolling_queue.popleft()  # Remove oldest frame
        frame_count += 1
        if frame_count >= 1000:
            frame_count = 0

    cap.release()


def initialize_threads(cap, frame_queue, rolling_queue, model_store, sp, gp, graphics):
    """
    Initializes and starts all necessary threads.

    Parameters:
    - cap (cv2.VideoCapture): The video capture source
    - frame_queue (queue.Queue): Queue for the latest frame
    - rolling_queue (queue.Queue): Queue for historical frame data
    - model_store (ModelStore): Model container for AI-based processing
    - sp (SpotifyPlayer): Handles music playback
    - gp (GPINFO): Manages game states
    """
    threads = [
        threading.Thread(target=update_frames, args=(cap, frame_queue, rolling_queue), daemon=True),
        threading.Thread(target=course_detect, args=(frame_queue, model_store, sp, gp), daemon=True),
        threading.Thread(target=state_detect, args=(frame_queue, model_store, sp, gp, rolling_queue, graphics), daemon=True),
        threading.Thread(target=run_stats, args=(frame_queue, rolling_queue, model_store, gp, graphics),daemon=True)
    ]

    for thread in threads:
        thread.start()


def main():
    """
    Main function to initialize and run the application
    """
    device = select_device()
    model_store = ModelStore(device = device)

    gp = GPINFO()
    sp = SpotifyPlayer()
    graphics = Graphics()
    cap = cv2.VideoCapture(0)

    # Thread-safe queues for frame handling
    frame_queue = queue.Queue(maxsize=1)
    rolling_queue = deque()

    # Start all background threads
    initialize_threads(cap, frame_queue, rolling_queue, model_store, sp, gp, graphics)

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

    # Clean up resources before exiting
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
     main()

