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
        out.write(frame)

    out.release()
    print(f"Saved highlight: {output_path}")


def scan_highlights(prediction: list, rolling_queue: queue.Queue[np.ndarray], history: list):
    timestamp = time.time()

    # Remove old history entries (> 5 seconds ago)
    history = [inst for inst in history if timestamp - inst[1] < 7]

    for instance in history:
        time_diff = timestamp - instance[4]
        if time_diff >= 5:
            for i in range(len(prediction)):
                place = prediction[i][1]
                if abs(place - instance[i]) >= 4:
                    rolling_frames = [f[0] for f in list(rolling_queue.queue)]
                    output_file = os.path.join(f"nextgenstats/highlights/highlight_{i}_{int(time.time())}.mp4")
                    save_video(rolling_frames, output_file, fps=30)
                    history.clear()  # Reset history after highlight
                    return history

    history.append((prediction, timestamp))
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
            history = scan_highlights(prediction, rolling_queue, history)
