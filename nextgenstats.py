import cv2
import queue
import random
import threading
import numpy as np
from __init__ import *
from typing import Optional


def calculate_highlightimportance(p0: int, p1: int) -> str:
    """
    Determines the importance of a highlight based on position changes
    """
    n0,n1 = (13-p0)**2, (13-p1)**2
    delta = abs(n1-n0)

    rankings = {
        12**2 - 5**2 : 'a', # 1 <-> 8, 119
        12**2 - 6**2 : 'b', # 1 <-> 7, 108
        12**2 - 7**2 : 'c',  # 1 <-> 6, 95
        12**2 - 8**2 : 'd',  # 1 <-> 5, 80
        12**2 - 9**2 : 'e',  # 1 <-> 4, 63
        12**2 - 8**2 : 'f',  # 2 <-> 5, 57
    }

    for threshold in rankings:
        if delta >= threshold:
            return rankings[threshold]

    if abs(p0-p1) >= 4:
        return 'g'
    return None


def save_video(frames, filename, graphics):
    """
    Saves frames to a video file.
    """
    if not frames:
        return

    fps = 30
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 format
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in frames:
        flipped_frame = cv2.flip(frame,0)
        out.write(flipped_frame)

    out.release()
    print(f"Saved highlight: {filename}")

def wrap_diff(a, b, max_value):
    return min(abs(a - b), max_value - abs(a - b))


def scan_highlights(prediction: list, rolling_queue: queue.Queue[np.ndarray], history: list, gp, frame_count, graphics):
    # Remove old history entries (> 14 seconds ago)
    fps = 30
    history = [inst for inst in history if wrap_diff(frame_count, inst[4], 1000) < (13*fps)]

    update = False
    for i in range(len(prediction)):
        place, confidence = prediction[i][1], prediction[i][2]
        if confidence >= 0.95:
            gp.places[i] = place
            update = True

    for instance in history:
        time_diff = wrap_diff(frame_count,instance[4], 1000)
        if time_diff >= (8*fps):
            for i in range(len(gp.places)):
                place = int(gp.places[i])
                p0, p1 = place, instance[i]
                rank = calculate_highlightimportance(place, instance[i])
                if rank:
                    rolling_frames = [
                        f[0] for f in rolling_queue
                        if ((f[1] >= instance[4]-(2*fps)) and (f[1] <= frame_count+(2*fps)))
                    ]
                    length = len(rolling_frames)
                    if length >= fps*4:
                        filename = os.path.join(f"nextgenstats/highlights/{rank}_{i}_{int(time.time())}_{p0}_{p1}_{length}.mp4")
                        save_video(rolling_frames, filename, graphics)
        else:
            break

    if update:
        new_instance = []
        for i in range(len(gp.places)):
            new_instance.append(int(gp.places[i]))

        history.append(new_instance + [frame_count])
    return history


def run_stats(frame_queue: queue.Queue[np.ndarray], rolling_queue: queue.Queue[np.ndarray], model_store: ModelStore,
              gp: GPINFO, graphics):
    #history = []

    while True:

        frame = frame_queue.get()
        if gp.course_state >= 1:
            labels = model_store.models['countdown_detector'].predict(frame)
            for i, (region, prediction, confidence) in enumerate(labels):
                if i == 0:
                    if prediction == 'GO' and confidence >= 0.98:
                        gp.course_state = 2
                if prediction == 'FINISH' and confidence >= 0.99 and gp.course_state == 2:
                    gp.course_state = 3

        #if gp.course_state == 2:
            #if rolling_queue:
                #(peak_frame, frame_count) = rolling_queue[-1]
                #prediction = model_store.models['placement_detector'].predict(peak_frame)
                #history = scan_highlights(prediction, rolling_queue, history, gp, frame_count, graphics)

