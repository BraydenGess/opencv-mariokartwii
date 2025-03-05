import cv2
import shutil
import queue
import random
import numpy as np
from __init__ import *
from typing import Optional

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
        clear_directory('nextgenstats/highlights')
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
            pass


def state_detect(frame_queue: queue.Queue[np.ndarray], model_store: ModelStore, sp: SpotifyPlayer, gp) -> bool:
    while True:
        frame = frame_queue.get()
        screen, confidence = model_store.models['state_detector'].predict(frame)
        if confidence >= 0.965:
            control(frame = frame, model_store = model_store,screen = screen, sp = sp, gp = gp)
