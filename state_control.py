import os
import cv2
import queue
import numpy as np

from __init__ import *
from nextgenstats import scan_highlights


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
    # Preprocess the frame for gray conversion and binary thresholding
    cropped_frame = frame[50:125, 510:650]
    gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, 128, 255, cv2.THRESH_BINARY)

    # List and load reference images only once
    reference_dir = 'production/referenceimages/player_counts'
    ref_files = [f for f in os.listdir(reference_dir) if f.endswith(".png")]

    ref_images = [
        (filename,cv2.threshold(cv2.imread(os.path.join(reference_dir, filename),
                                cv2.IMREAD_GRAYSCALE)[50:125, 510:650], 128, 255, cv2.THRESH_BINARY)[1])
        for filename in os.listdir(reference_dir) if filename.endswith(".png")
    ]

    # Match each reference image with the frame
    for filename, ref_image in ref_images:
        result = cv2.matchTemplate(binary_frame, ref_image, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        # Update player count if match found with high confidence
        if max_val >= 0.925:
            gp.player_count = int(filename[0])
            break


def vehicle_detect(frame: np.ndarray, model_store: ModelStore, gp: GPINFO) -> None:
    """
    Detect vehicles in the given frame and update vehicles in GPINFO
    """
    predictions = model_store.models['vehicle_detector'].predict(frame = frame, players = gp.player_count)
    for i, (region, vehicle_name, confidence) in enumerate(predictions):
        if confidence >= 0.95:
            gp.vehicles[i] = vehicle_name


def character_detect(frame: np.ndarray, model_store: ModelStore, gp: GPINFO) -> None:
    """
    Detect characters in the given frame and update GPINFO
    """
    predictions = model_store.models['character_detector'].predict(frame = frame, players = gp.player_count)
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
        gp.course_history.clear()
        gp.clear_directory(directory_path = 'graphics/assets/highlights')

        if sp.course_queued != "Opening":
            sp.queue_newsong(course_name = "Opening")

    # Process screens only if the main state is valid
    if gp.main_state >= 0:
        screen_actions = {
            'players': lambda: player_count(frame, gp),
            'characters': lambda: (setattr(gp, 'main_state', 1),character_detect(frame, model_store, gp)),
            'vehicles': lambda: (setattr(gp, 'main_state', 2), vehicle_detect(frame, model_store, gp)),
        }

        action = screen_actions.get(screen)
        if action:
            action()


def state_detect(frame_queue: queue.Queue[np.ndarray], model_store: ModelStore, sp: SpotifyPlayer, gp: GPINFO,
                 rolling_queue, graphics) -> None:
    """
    Process frames and detect whether state screen is detected
    """
    history = []
    while True:
        frame = frame_queue.get()
        screen, confidence = model_store.models['state_detector'].predict(frame)
        if confidence >= 0.965:
            control(frame = frame, model_store = model_store, screen = screen, sp = sp, gp = gp)

        ### NextGenStats capability, high I/O so moved here for better thread allocating, fix later
        if gp.course_state == 2:
            if rolling_queue:
                (peak_frame, frame_count) = rolling_queue[-1]
                prediction = model_store.models['placement_detector'].predict(frame = peak_frame, players = gp.player_count)
                history = scan_highlights(prediction, rolling_queue, history, gp, frame_count, graphics)
