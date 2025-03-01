import cv2
import queue
import random
import numpy as np
from __init__ import *
from typing import Optional

def pause_toggle(screen: str, sp: SpotifyPlayer) -> None:
    if screen == 'homescreen':
        if not sp.is_paused:
            sp.pause()
    else:
        if sp.is_paused:
            sp.resume()


def control(screen: str, sp: SpotifyPlayer, state_trigger: bool) -> bool:
    pause_toggle(screen = screen, sp = sp)

    if screen == 'main':
        state_trigger = True
        if sp.course_queued != "Opening":
            sp.queue_newsong(course_name = "Opening")
        return state_trigger

    # state_detect controls whether the models check.
    # Should only be valid states if after main is detected and before startrace closes the menu
    if state_detect:
        if screen == 'characters':
            # do character stuff
            pass
        elif screen == 'vehicles':
            # do vehicle stuff
            pass
        elif screen == 'startrace':
            state_trigger = False
    return state_trigger


def state_detect(frame_queue: queue.Queue[np.ndarray], model_store: ModelStore,
                                                    sp: SpotifyPlayer, state_trigger: bool) -> bool:
    while True:
        frame = frame_queue.get()
        screen, confidence = model_store.models['state_detector'].predict(frame)
        if confidence >= 0.995:
            state_trigger = control(screen = screen, sp = sp, state_trigger = state_trigger)
