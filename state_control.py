import cv2
import shutil
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

def vehicle_detect(frame, model_store, gp):
    predictions = model_store.models['vehicle_detector'].predict(frame)
    for i in range(len(predictions)):
        region = predictions[i]
        if region[2] >= 0.95:
            gp.vehicles[i] = region[1]


def character_detect(frame, model_store, gp):
    predictions = model_store.models['character_detector'].predict(frame)
    for i in range(len(predictions)):
        region = predictions[i]
        if region[2] >= 0.95:
            gp.characters[i] = region[1]


def control(frame, model_store, screen: str, sp: SpotifyPlayer, gp) -> bool:
    pause_toggle(screen = screen, sp = sp)

    if screen == 'main':
        gp.main_state = 0
        gp.clear_directory(directory_path = 'nextgenstats/highlights')
        if sp.course_queued != "Opening":
            sp.queue_newsong(course_name = "Opening")

    # state_detect controls whether the models check.
    # Should only be valid states if after main is detected and before startrace closes the menu
    if gp.main_state >= 0:
        if screen == 'characters':
            gp.main_state = 1
            character_detect(frame, model_store, gp)
        elif screen == 'vehicles':
            gp.main_state = 2
            vehicle_detect(frame, model_store, gp)


def state_detect(frame_queue: queue.Queue[np.ndarray], model_store: ModelStore, sp: SpotifyPlayer, gp) -> bool:
    while True:
        frame = frame_queue.get()
        screen, confidence = model_store.models['state_detector'].predict(frame)
        if confidence >= 0.965:
            control(frame = frame, model_store = model_store, screen = screen, sp = sp, gp = gp)
