import cv2
import random
import numpy as np
from __init__ import *
from typing import Optional

def pause_toggle(frame):
    #pause = predict(frame)
    # if pause:
        #pause
    # if not pause:
        # don't pause
    pass


def play_music(frame: np.ndarray, model_store: ModelStore, sp: SpotifyPlayer):
    course_name, confidence, text_detections = model_store.models['course_detector'].detect_course(frame)
    if course_name:
        if course_name != sp.course_queued:
            sp.queue_newsong(course_name)
            num = random.randint(1,100)
            cv2.imwrite(f'development/triggers/{course_name}_{num}.png', frame)


def run_audio(frame: np.ndarray, model_store: ModelStore, sp: SpotifyPlayer):
    pause_toggle(frame)
    play_music(frame = frame, model_store = model_store, sp = sp)
