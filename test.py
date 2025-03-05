import os
import cv2
import time
import queue
import threading
import random

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

def get_highlights(frame_queue: queue.Queue, rolling_queue: queue.Queue):
    """Detects highlights based on prediction changes."""
    history = []

    while True:
        if frame_queue.empty():
            time.sleep(0.1)
            continue

        frame = frame_queue.get()
        timestamp = time.time()
        prediction = random.randint(1, 12)

        # Remove old history entries (> 5 seconds ago)
        history = [inst for inst in history if timestamp - inst[1] < 5]

        for instance in history:
            time_diff = timestamp - instance[1]
            if time_diff >= 3 and abs(prediction - instance[0]) >= 4:
                print(f"Highlight detected! Prediction diff: {abs(prediction - instance[0])}, Time diff: {time_diff}")

                # Save rolling frames
                rolling_frames = [f[0] for f in list(rolling_queue.queue)]
                output_file = os.path.join(f"nextgenstats/highlights/highlight_{int(time.time())}.mp4")
                save_video(rolling_frames, output_file, fps=30)

                history.clear()  # Reset history after highlight
                break

        history.append((prediction, timestamp))

def update_frames(cap: cv2.VideoCapture, frame_queue: queue.Queue, rolling_queue: queue.Queue):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        prev_time, fps = None, 30
    else:
        frame_duration = 1 / fps
    while cap.isOpened():
        ret, new_frame = cap.read()
        if not ret:
            break
        timestamp = time.time() if fps != 0 else prev_time + frame_duration if prev_time else time.time()
        if fps == 0:
            if prev_time:
                frame_duration = timestamp - prev_time
                fps = 1 / frame_duration
            prev_time = timestamp
        max_rolling_frames = int(fps * 5)
        if not frame_queue.empty():
            frame_queue.get_nowait()
        frame_queue.put(new_frame)
        rolling_queue.put((new_frame, timestamp))
        while rolling_queue.qsize() > max_rolling_frames:
            rolling_queue.get()
    cap.release()


def main():
    cap = cv2.VideoCapture(0)
    frame_queue = queue.Queue(maxsize=1)
    rolling_queue = queue.Queue()

    frame_thread = threading.Thread(target = update_frames, args = (cap, frame_queue, rolling_queue), daemon = True)
    highlight_thread = threading.Thread(target = get_highlights, args = (frame_queue, rolling_queue), daemon = True)

    frame_thread.start()
    highlight_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
