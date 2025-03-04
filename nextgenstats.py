import cv2
import queue
import random
import numpy as np
from __init__ import *
from typing import Optional


def run_stats(frame_queue: queue.Queue[np.ndarray], model_store: ModelStore, sp: SpotifyPlayer, gp) -> bool:
    while True:
        frame = frame_queue.get()
        if gp.course_state == 1:

            # detect for start
            # if started turn to 2
            # if finished turn to 3
            pass
        if gp.course_state == 2:
            predictions = model_store.models['placement_detector'].predict(frame)
            # detect placement
            # detect finish
            pass