import os
import cv2
import queue
import random
import shutil
import numpy as np
from typing import Optional

from __init__ import *

def clear_directory(directory):
    """Deletes all files and subdirectories inside a directory."""
    if not os.path.exists(directory):
        return
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # Delete file
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Delete folder and its contents
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")


def course_detect(frame_queue: queue.Queue[np.ndarray], model_store: ModelStore, sp: SpotifyPlayer, gp : GPINFO):
    while True:
        frame = frame_queue.get()
        course_name, confidence, text_detection = model_store.models['course_detector'].detect_course(frame)
        if course_name:
            if course_name != sp.course_queued:
                sp.queue_newsong(course_name)

                gp.main_state = -1
                gp.course_state = 1
                clear_directory("nextgenstats/highlights")

                ### Record what triggered it for model fine-tuning
                num = random.randint(1,100)
                cv2.imwrite(f'development/triggers/{course_name}_{num}.png', frame)