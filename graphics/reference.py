import os
import cv2
import time

def capture_frames(save_path="frames/", duration=3, fps=30):
    """Capture frames from OpenCV and save them as image files."""
    cap = cv2.VideoCapture(0)  # Open webcam
    os.makedirs(save_path, exist_ok=True)  # Ensure directory exists

    frame_count = 0
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame as image file
        frame_filename = os.path.join(save_path, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Captured {frame_count} frames and saved them to {save_path}")

import pygame
import numpy as np

def load_frames_from_disk(folder_path="frames/"):
    """Loads saved image frames from disk into a list."""
    frame_files = sorted(os.listdir(folder_path))  # Ensure correct order
    frames = []

    for file in frame_files:
        file_path = os.path.join(folder_path, file)
        frame = cv2.imread(file_path)
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frames.append(frame)

    print(f"Loaded {len(frames)} frames from {folder_path}")
    return frames

def play_movie(display_surface, frames, fps=30):
    """Plays saved frames on the Pygame window."""
    if not frames:
        print("Error: No frames to display!")
        return

    clock = pygame.time.Clock()
    frame_index = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

        # Convert OpenCV frame to Pygame surface
        frame_surface = pygame.surfarray.make_surface(np.rot90(frames[frame_index], 3))
        display_surface.blit(frame_surface, (0, 0))
        pygame.display.update()

        frame_index = (frame_index + 1) % len(frames)  # Loop through frames
        clock.tick(fps)


class Graphics():
    def __init__(self, screen_setting='fullscreen'):
        self.screen_setting = screen_setting
        self.x, self.y = None, None
        self.display_surface = None
        self.frames = []  # Store frames
        self.setup()
        self.capture_frames()  # Capture frames before running

    def setup(self):
        pygame.init()
        screen = pygame.display.set_mode()
        self.x, self.y = screen.get_size()
        flags = pygame.FULLSCREEN if self.screen_setting.lower() == 'fullscreen' else 0
        self.display_surface = pygame.display.set_mode((self.x, self.y), flags)
        pygame.display.set_caption("OpenCV MarioKart")

    def capture_frames(self):
        """Capture and save frames before running the game."""
        capture_frames()  # Save frames to disk
        self.frames = load_frames_from_disk()  # Load them into memory

    def run(self):
        running = True
        t1 = time.time()

        while running:
            self.display_surface.fill((0, 0, 0))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False  # Quit if Q or ESC is pressed

            if time.time() - t1 > 5:
                play_movie(self.display_surface, self.frames)  # Play recorded frames

            pygame.display.update()


if __name__ == "__main__":
    game = Graphics()
    game.run()

