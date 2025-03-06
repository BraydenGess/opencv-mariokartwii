import cv2
import queue
import random
import numpy as np
from __init__ import *
from typing import Optional


def calculate_highlightimportance(p0: int,p1: int) -> str:
    ### Counts place changes giving more weight to being higher up
    n0,n1 = (13-p0)**2, (13-p1)**2
    delta = abs(n1-n0)
    if delta >= (12**2 - 5**2): # 1 <-> 8, 119
        return 'a'
    elif delta >= (12**2 - 6**2): # 1 <-> 7, 108
        return 'b'
    elif delta >= (12**2 - 7**2): # 1 <-> 6, 95
        return 'c'
    elif delta >= (12**2 - 8**2): # 1 <-> 5, 80
        return 'd'
    elif delta >= (12**2 - 9**2): # 1 <-> 4, 63
        return 'e'
    elif delta >= (11**2 - 8**2): # 2 <-> 5, 57
        return 'f'
    elif abs(p0-p1) >= 4:
        return 'g'
    return False


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
    org_timestamp = time.time()
    # Remove old history entries (> 12 seconds ago)
    history = [inst for inst in history if org_timestamp - inst[4] < 14]
    update = False

    for i in range(len(prediction)):
        place, confidence = prediction[i][1], prediction[i][2]
        if confidence >= 0.95:
            gp.places[i] = place
            update = True

    cur_timestamp = time.time()
    for instance in history:
        time_diff = cur_timestamp - instance[4]
        if time_diff >= 10:
            for i in range(len(gp.places)):
                place = int(gp.places[i])
                p0, p1 = place, instance[i]
                rank = calculate_highlightimportance(place, instance[i])
                if rank:
                    rolling_frames = [f[0] for f in list(rolling_queue.queue)[:int(len(rolling_queue.queue) * 0.85)]]
                    output_file = os.path.join(f"nextgenstats/highlights/{rank}_{i}_{int(time.time())}_{p0}_{p1}.mp4")
                    save_video(rolling_frames, output_file, fps=24)
                    #history.clear()
                    return history
        else:
            break

    if update:
        new_instance = []
        for i in range(len(gp.places)):
            new_instance.append(int(gp.places[i]))

        history.append(new_instance + [org_timestamp])
    return history


def run_stats(frame_queue: queue.Queue[np.ndarray], rolling_queue: queue.Queue[np.ndarray], model_store: ModelStore,
              sp: SpotifyPlayer, gp) -> bool:
    history = []
    while True:
        frame = frame_queue.get()
        if gp.course_state >= 1:
            labels = model_store.models['countdown_detector'].predict(frame)
            for i in range(len(labels)):
                [region, prediction, confidence] = labels[i]
                if i == 0:
                    if confidence >= 0.98:
                        if prediction == 'GO':
                            gp.course_state = 2
                if prediction == 'FINISH':
                    if confidence >= 0.99:
                        if gp.course_state == 2:
                            gp.course_state = 3
        if gp.course_state == 2:
            prediction = model_store.models['placement_detector'].predict(frame)
            history = scan_highlights(prediction, rolling_queue, history, gp)
