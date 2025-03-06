import os
import cv2
import sys
import pygame
import numpy as np
import os
import re


def parse_filename(filename):
    """Extracts letter, number, and timestamp from the filename."""
    match = re.match(r"([a-zA-Z])_(\d+)_(\d+)_.*\.mp4", filename)
    if match:
        letter, number, timestamp = match.groups()
        return letter, int(number), int(timestamp), filename
    return None


def load_highlight_videos():
    HIGHLIGHT_DIR = "nextgenstats/highlights"
    """Loads highlight videos and removes files if a higher letter or newer same letter exists within 3 seconds."""
    if not os.path.exists(HIGHLIGHT_DIR):
        os.makedirs(HIGHLIGHT_DIR, exist_ok=True)
        return []

    highlight_files = [f for f in os.listdir(HIGHLIGHT_DIR) if f.endswith(".mp4")]

    grouped_files = {}

    # Step 1: Parse and group by number
    for file in highlight_files:
        parsed = parse_filename(file)
        if parsed:
            letter, number, timestamp, filename = parsed
            grouped_files.setdefault(number, []).append((letter, timestamp, filename))

    selected_files = []

    # Step 2: Process each group
    for key in grouped_files:
        # Sort by letter (ascending) and timestamp (descending)
        grouped_files[key].sort(key=lambda x: (x[0], -x[1]))

        kept_files = []

        for file in grouped_files[key]:
            letter, timestamp, filename = file
            should_keep = True

            for kept in kept_files:
                kept_letter, kept_timestamp, kept_filename = kept

                # If a higher letter exists within 3 seconds, remove this file
                if kept_letter > letter and abs(kept_timestamp - timestamp) <= 12:
                    should_keep = False
                    break

                # If the same letter exists within 3 seconds and is newer, remove this file
                if kept_letter == letter and abs(kept_timestamp - timestamp) <= 12:
                    should_keep = False
                    break

            if should_keep:
                kept_files.append(file)
            else:
                os.remove(os.path.join(HIGHLIGHT_DIR, filename))  # Delete unnecessary file

        # Collect final kept files
        selected_files.extend([os.path.join(HIGHLIGHT_DIR, f[2]) for f in kept_files])

    return selected_files


def play_video(display_surface, video_path, x, y):
    """Plays a video file on the Pygame window."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}!")
        return

    fps = 30
    clock = pygame.time.Clock()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop when video ends

        player = int(video_path.split('_')[1])
        marg = 12
        frame = frame[:,marg:len(frame[1])-(marg*2)]
        new_x = x - (3*marg)
        if player == 0:
            frame = frame[y//2:, :new_x//2]
        if player == 1:
            frame = frame[y//2:, new_x//2:]
        if player == 2:
            frame = frame[:y//2, marg:new_x // 2]
        if player == 3:
            frame = frame[:y//2, new_x // 2:]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frame = cv2.resize(frame, (x, y))
        frame_surface = pygame.surfarray.make_surface(np.rot90(frame, 3))
        display_surface.blit(frame_surface, (0, 0))
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    cap.release()
                    return  # Quit playback

        clock.tick(fps)

    cap.release()


def play_highlights(display_surface, gp, x, y):
    """Loop through all highlight videos until gp.course_state is not 3."""
    highlight_videos = load_highlight_videos()
    if not highlight_videos:
        return

    while gp.course_state == 3:
        for video_path in highlight_videos:
            if ((gp.course_state != 3) or (gp.main_state == 0)):
                break  # Exit the loop if course_state is no longer 3
            play_video(display_surface, video_path, x, y)  # Play current highlight
        return
    return