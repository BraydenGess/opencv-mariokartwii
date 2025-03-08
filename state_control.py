import os
import cv2
import queue
import numpy as np

from __init__ import *


def pause_toggle(screen: str, sp: SpotifyPlayer) -> None:
    """
    Toggle Spotify pause state based on the current screen
    """
    if screen == 'homescreen':
        if not sp.is_paused:
            sp.pause()
    else:
        if sp.is_paused:
            sp.resume()


def player_count(frame: np.ndarray, gp: GPINFO) -> None:
    """
    Determines the number of players by matching a cropped region of the frame against reference images

    Updates `gp.player_count` if a match is found with high confidence.
    """

    gray_frame = cv2.cvtColor(frame[50:125, 510:650], cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, 128, 255, cv2.THRESH_BINARY)

    REFERENCE_DIR = 'production/referenceimages/player_counts'
    ref_files = [f for f in os.listdir(REFERENCE_DIR) if f.endswith(".png")]
    ref_images = list()
    for filename in ref_files:
        ref_img = cv2.imread(os.path.join(REFERENCE_DIR,filename), cv2.IMREAD_GRAYSCALE)
        if ref_img is not None:
            _, ref_img = cv2.threshold(ref_img[50:125, 510:650], 128, 255, cv2.THRESH_BINARY)
            ref_images.append((filename, ref_img))

    for filename, ref_image in ref_images:
        result = cv2.matchTemplate(binary_frame, ref_image, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val >= 0.95:
            gp.player_count = int(filename[0])


def vehicle_detect(frame: np.ndarray, model_store: ModelStore, gp: GPINFO) -> None:
    """
    Detect vehicles in the given frame and update vehicles in GPINFO
    """
    predictions = model_store.models['vehicle_detector'].predict(frame)
    for i, (region, vehicle_name, confidence) in enumerate(predictions):
        if confidence >= 0.95:
            gp.vehicles[i] = vehicle_name


def character_detect(frame: np.ndarray, model_store: ModelStore, gp: GPINFO) -> None:
    """
    Detect characters in the given frame and update GPINFO
    """
    predictions = model_store.models['character_detector'].predict(frame)
    for i, (region, character_name, confidence) in enumerate(predictions):
        if confidence >= 0.95:
            gp.characters[i] = character_name


def control(frame, model_store, screen: str, sp: SpotifyPlayer, gp: GPINFO) -> None:
    """
    Control game states based on which main screen is detected
    """
    pause_toggle(screen = screen, sp = sp)

    if screen == 'main':
        gp.main_state = 0
        gp.clear_directory(directory_path = 'nextgenstats/highlights')
        if sp.course_queued != "Opening":
            sp.queue_newsong(course_name = "Opening")

    # Should only be valid states if after main is detected
    if gp.main_state >= 0:
        if screen == 'players':
            player_count(frame, gp)
        if screen == 'characters':
            gp.main_state = 1
            character_detect(frame, model_store, gp)
        elif screen == 'vehicles':
            gp.main_state = 2
            vehicle_detect(frame, model_store, gp)


def state_detect(frame_queue: queue.Queue[np.ndarray], model_store: ModelStore, sp: SpotifyPlayer, gp: GPINFO) -> None:
    """
    Process frames and detect whether state screen is detected
    """
    while True:
        frame = frame_queue.get()
        screen, confidence = model_store.models['state_detector'].predict(frame)
        if confidence >= 0.965:
            control(frame = frame, model_store = model_store, screen = screen, sp = sp, gp = gp)
