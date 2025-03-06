import os
import cv2
import time
import queue
import random
import shutil
import numpy as np
from typing import Optional

from __init__ import *


def course_detect(frame_queue: queue.Queue[np.ndarray], model_store: ModelStore, sp: SpotifyPlayer, gp : GPINFO):
    """
    Continuously process frames from a queue to detect a course name, trigger song, and save frame for fine-tuning

    Parameters:
    - frame_queue (queue.Queue[np.ndarray]): A thread-safe queue containing frames (images) for processing
    - model_store (ModelStore): An object storing models, including the course detector
    - sp (SpotifyPlayer): The Spotify player instance responsible for handling music playback
    - gp (GPINFO): An object managing game states and clearing directories.
    """
    while True:
        try:
            # Get frame from queue (blocking)
            frame = frame_queue.get()

            # Detect course
            course_name, confidence, text_detection = model_store.models['course_detector'].detect_course(frame)

            if course_name and course_name != sp.course_queued:
                sp.queue_newsong(course_name)

                # Manage Game States
                gp.main_state = -1
                gp.course_state = 1
                gp.clear_directory(directory_path = "nextgenstats/highlights")

                # Record what triggered it for model fine-tuning
                timestamp = int(time.time())
                cv2.imwrite(f'development/triggers/{course_name}_{timestamp}.png', frame)

        except queue.Empty:
            # Avoid busy waiting if using non-blocking get()
            time.sleep(0.01)
        except Exception as e:
            print(f"[ERROR] course_detect encountered an issue: {e}")