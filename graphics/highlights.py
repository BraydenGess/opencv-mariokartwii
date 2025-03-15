import os
import cv2
import sys
import pygame
import numpy as np
import os
import re


def parse_filename(file):
    """
    Parses filenames like 'a_0_2023_4_9_12.mp4' and returns (letter, number, timestamp, filename).
    """
    try:
        parts = file[:-4].split("_")  # Remove .mp4 and split
        letter = parts[0]
        number = int(parts[1])
        timestamp = int(parts[2])  # Assuming timestamp is a single integer
        return letter, number, timestamp, file
    except (IndexError, ValueError):
        return None


def load_highlight_videos():
    """
    Loads highlight videos and removes files if a higher letter or newer same letter exists within 3 seconds.
    """
    HIGHLIGHT_DIR = "graphics/assets/highlights"
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
    for number in sorted(grouped_files.keys()):  # Ensure 'a_0' is first, then 'b_2'
        # Sort by letter (ascending) and timestamp (descending)
        grouped_files[number].sort(key=lambda x: (x[0], -x[1]))

        kept_files = []

        for file in grouped_files[number]:
            letter, timestamp, filename = file
            should_keep = True

            for kept in kept_files:
                kept_letter, kept_timestamp, kept_filename = kept

                # If a higher letter exists within 3 seconds, remove this file
                if kept_letter > letter and abs(kept_timestamp - timestamp) <= 3:
                    should_keep = False
                    break

                # If the same letter exists within 3 seconds and is newer, remove this file
                if kept_letter == letter and abs(kept_timestamp - timestamp) <= 3:
                    should_keep = False
                    break

            if should_keep:
                kept_files.append(file)
            else:
                os.remove(os.path.join(HIGHLIGHT_DIR, filename))  # Delete unnecessary file

        # Collect final kept files
        selected_files.extend([os.path.join(HIGHLIGHT_DIR, f[2]) for f in kept_files])

    return selected_files


def play_video(graphics, display_surface, video_path, x, y, gp):
    """Plays a video file on the Pygame window."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}!")
        return

    fps = 30
    clock = pygame.time.Clock()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or gp.course_state != 3:
            break  # Stop when video ends

        region = int(video_path.split('_')[1])
        if gp.player_count == 4:
            key = f'region{region}'
            [y0, y1, x0, x1] = graphics.regions_4[key]
        elif gp.player_count == 2:
            key = f'region{region}'
            [y0, y1, x0, x1] = graphics.regions_2[key]

        frame = frame[int(y0):int(y1), int(x0):int(x1)]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frame = cv2.resize(frame, (x, y))
        frame = cv2.flip(frame, 1)
        frame_surface = pygame.surfarray.make_surface(np.rot90(frame))
        display_surface.blit(frame_surface, (0, 0))

        txt, txtRect = graphics.create_text('Arial', 32, video_path, (255, 255, 255),
         (graphics.x*3//4, graphics.y//4), 'center')
        display_surface.blit(txt, txtRect)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    gp.course_state = 1
                    cap.release()
                    return  # Quit playback

        clock.tick(fps)
    cap.release()

def play_top(graphics, display_surface, label, gp):
    """Plays a video file on the Pygame window."""
    x,y = graphics.x, graphics.y
    video_path = f'graphics/assets/Top10/{label}.mp4'
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}!")
        return
    fps = 30
    clock = pygame.time.Clock()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or gp.course_state != 3:
            break  # Stop when video ends
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frame = cv2.resize(frame, (x, y))
        frame = cv2.flip(frame, 1)
        frame_surface = pygame.surfarray.make_surface(np.rot90(frame))
        display_surface.blit(frame_surface, (0, 0))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    gp.course_state = 1
                    cap.release()
                    return  # Quit playback
        clock.tick(fps)
    cap.release()

def play_highlights(graphics, display_surface, gp, x, y):
    """Loop through all highlight videos until gp.course_state is not 3."""
    highlight_videos = load_highlight_videos()[:2]
    if not highlight_videos:
        return
    ### Intro video
    label = 'Intro'
    play_top(graphics, display_surface, label, gp)

    while gp.course_state == 3:
        for i, video_path in enumerate(highlight_videos):
            if ((gp.course_state != 3) or (gp.main_state == 0)):
                return

            play_top(graphics, display_surface, str(i+1), gp)
            play_video(graphics, display_surface, video_path, x, y, gp)  # Play current highlight
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE)):
                    gp.course_state = 0  # Ensure the loop stops
                    return