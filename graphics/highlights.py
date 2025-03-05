import os
import cv2
import sys
import pygame
import numpy as np

def load_highlight_videos():
    """Finds all highlight videos in the directory, sorted by modification time."""
    HIGHLIGHT_DIR = 'nextgenstats/highlights'
    if not os.path.exists(HIGHLIGHT_DIR):
        os.makedirs(HIGHLIGHT_DIR, exist_ok=True)
        return []

    highlight_files = sorted(
        [f for f in os.listdir(HIGHLIGHT_DIR) if f.endswith(".mp4")],
        key=lambda x: os.path.getmtime(os.path.join(HIGHLIGHT_DIR, x)),
        reverse=True
    )
    return [os.path.join(HIGHLIGHT_DIR, f) for f in highlight_files]


def play_video(display_surface, video_path, x, y):
    """Plays a video file on the Pygame window."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    clock = pygame.time.Clock()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop when video ends

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

    for video_path in highlight_videos:
        while gp.course_state == 3:
            play_video(display_surface, video_path, x, y)  # Play current highlight
            if ((gp.course_state != 3) or (gp.main_state == 0)):
                break  # Exit the loop if course_state is no longer 3