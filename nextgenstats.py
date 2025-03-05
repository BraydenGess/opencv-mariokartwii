import cv2
import queue
import random
import numpy as np
from __init__ import *
from typing import Optional

def save_video(frames, output_path, fps):
    """Saves frames to a video file."""
    if not frames:
        return

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        flipped_frame = cv2.flip(frame, 0)
        out.write(flipped_frame)

    out.release()
    print(f"Saved highlight: {output_path}")


def scan_highlights(prediction: list, rolling_queue: queue.Queue[np.ndarray], history: list, gp):
    timestamp = time.time()

    # Remove old history entries (> 5 seconds ago)
    history = [inst for inst in history if timestamp - inst[4] < 12]

    for i in range(len(prediction)):
        place, confidence = prediction[i][1], prediction[i][2]
        if confidence >= 0.95:
            gp.places[i] = place

    for instance in history:
        time_diff = timestamp - instance[4]
        if time_diff >= 8:
            for i in range(len(gp.places)):
                place = int(gp.places[i])
                if abs(place - instance[i]) >= 3:
                    rolling_frames = [f[0] for f in list(rolling_queue.queue)]
                    output_file = os.path.join(f"nextgenstats/highlights/highlight_{i}_{int(time.time())}.mp4")
                    save_video(rolling_frames, output_file, fps=30)
                    history.clear()
                    return history

    new_instance = []
    for i in range(len(gp.places)):
        new_instance.append(int(gp.places[i]))

    history.append(new_instance + [timestamp])
    return history


def run_stats(frame_queue: queue.Queue[np.ndarray], rolling_queue: queue.Queue[np.ndarray], model_store: ModelStore,
              sp: SpotifyPlayer, gp) -> bool:
    history = []
    while True:
        frame = frame_queue.get()
        if gp.course_state >= 1:
            prediction, confidence = model_store.models['countdown_detector'].predict(frame)
            if confidence > 0.98:
                if prediction == 'GO':
                    gp.course_state = 2
                if prediction == 'FINISH':
                    gp.course_state = 3
        if gp.course_state == 2:
            prediction = model_store.models['placement_detector'].predict(frame)
            history = scan_highlights(prediction, rolling_queue, history, gp)
