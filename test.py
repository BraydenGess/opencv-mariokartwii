import cv2
import sys
import torch
import cv2
import queue
import pygame
import threading
import collections
from collections import deque
from urllib.request import urlopen


from __init__ import *
from spotify_audio import course_detect
from state_control import state_detect
from nextgenstats import run_stats
from graphics.graphics import Graphics
#print(0%2)
#exit()
#frame = cv2.imread('development/Images/Placement/12+11+7+10_MoonviewHighway_123.png')
#height, width, _ = frame.shape
#margin = width//48
#print(width, height)
#frame = frame[:height//2-margin,margin:width//2-margin]
#frame = frame[:height//2-margin,width//2+margin:width-margin]

#cv2.imshow('Cropped Image', frame)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

def update_frames(cap: cv2.VideoCapture, frame_queue: queue.Queue, rolling_queue: collections.deque):
    """"
    Continuously reads frames from a video capture source and updates the frame queues

    Parameters:
    - cap (cv2.VideoCapture): The video capture source
    - frame_queue (queue.Queue): A queue holding the latest captured frame
    - rolling_queue (queue.Queue): A queue storing frames with timestamps for historical reference
    """
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    buffer_seconds = 24
    max_rolling_frames = int(fps * buffer_seconds)
    frame_count = 0

    while cap.isOpened():
        ret, new_frame = cap.read()
        if not ret:
            break

        # Keep only the latest frame in frame_queue
        if not frame_queue.empty():
            frame_queue.get_nowait()
        frame_queue.put(new_frame)

        # Maintain rolling queue buffer
        rolling_queue.append((new_frame, frame_count))
        if len(rolling_queue) > max_rolling_frames:
            rolling_queue.popleft()  # Remove oldest frame
        frame_count += 1
        if frame_count >= 1000:
            frame_count = 0

    cap.release()

def main():
    gp = GPINFO()
    sp = SpotifyPlayer()
    graphics = Graphics()
    gp.main_state = -1
    gp.course_state = 2
    gp.course_history = []
    gp.characters = ['Funky Kong','Luigi','Yoshi','Peach']
    sp.course_queued = "Luigi Circuit"
    gp.course_history.append(sp.course_queued)
    img_str = 'https://i.scdn.co/image/ab67616d0000b2738399047ff71200928f5b6508'
    sp.song_queued = Song(song_name = "Thunderstruck", uri = 'spotify:track:7snQQk1zcKl8gZ92AnueZW',
                          img = img_str)

    sp.song_img = urlopen(img_str).read()
    gp.course_start = time.time()
    gp.player_count = 4
    #gp.characters[3] = 'Funky Kong'
    #gp.vehicles[3] = 'Flame Runner'
    #gp.player_count = 3
    cap = cv2.VideoCapture(1)

    frame_queue = queue.Queue(maxsize=1)
    rolling_queue = deque()

    frame_thread = threading.Thread(target=update_frames, args=(cap, frame_queue, rolling_queue), daemon=True)
    frame_thread.start()

    try:
        graphics.run(sp, gp, rolling_queue)  # This contains the event loop now
    except Exception as e:
        import traceback
        print("Error in graphics loop:", e)
        traceback.print_exc()  # Print the full error traceback
        pygame.quit()
        sys.exit()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")

    # Clean up resources before exiting
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
