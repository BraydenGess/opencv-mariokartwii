import cv2
import queue
import random
import numpy as np
from typing import Optional

from __init__ import *


def course_detect(frame_queue: queue.Queue[np.ndarray], model_store: ModelStore, sp: SpotifyPlayer):
    while True:
        frame = frame_queue.get()
        course_name, confidence, text_detection = model_store.models['course_detector'].detect_course(frame)
        if course_name:
            if course_name != sp.course_queued:
                sp.queue_newsong(course_name)
                num = random.randint(1,100)
                cv2.imwrite(f'development/triggers/{course_name}_{num}.png', frame)