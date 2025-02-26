import cv2
import random
import numpy as np
from __init__ import *
from typing import Optional

def control(screen: str, sp: SpotifyPlayer, state_detect: bool) -> bool:
    # state_detect controls whether the models check.
    # Should only be valid states if after main is detected and before startrace closes the menu
    if screen == 'main':
        state_detect = True
        if sp.course_queued != "Opening":
            sp.queue_newsong(course_name = "Opening")
        return state_detect

    if state_detect:
        if screen == 'characters':
            # do character stuff
            pass
        elif screen == 'vehicles':
            # do vehicle stuff
            pass
        elif screen == 'startrace':
            state_detect = False
    return state_detect


def run_statecontrol(frame: np.ndarray, model_store: ModelStore, sp: SpotifyPlayer, state_detect: bool) -> bool:
    screen, confidence = model_store.models['state_detector'].predict(frame)
    if confidence >= 0.995:
        state_detect = control(screen = screen, sp = sp, state_detect = state_detect)
    return state_detect
