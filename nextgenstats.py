import cv2
import queue
import random
import numpy as np
from __init__ import *
from typing import Optional


def run_stats(frame_queue: queue.Queue[np.ndarray], model_store: ModelStore, sp: SpotifyPlayer, gp) -> bool:
    while True:
        frame = frame_queue.get()
        if gp.course_state >= 1:
            prediction, confidence = model_store.models['countdown_detector'].predict(frame)
            if confidence > 0.99:
                if prediction == 'GO':
                    gp.course_state = 2
                if prediction == 'FINISH':
                    gp.course_state = 3
        if gp.course_state == 2:
            predictions = model_store.models['placement_detector'].predict(frame)
            #run highlight algo
            pass