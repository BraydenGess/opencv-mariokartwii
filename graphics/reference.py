import os
import cv2
import time
import pygame
import numpy as np

HIGHLIGHT_DIR = "nextgenstats/highlights"

def load_latest_highlight():
    """Finds the latest saved highlight video in the directory."""
    if not os.path.exists(HIGHLIGHT_DIR):
        os.makedirs(HIGHLIGHT_DIR, exist_ok=True)
        return None

    highlight_files = sorted(
        [f for f in os.listdir(HIGHLIGHT_DIR) if f.endswith(".mp4")],
        key=lambda x: os.path.getmtime(os.path.join(HIGHLIGHT_DIR, x)),
        reverse=True
    )

    return os.path.join(HIGHLIGHT_DIR, highlight_files[0]) if highlight_files else None

def play_video(display_surface, video_path):
    """Plays a video file on the Pygame window."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    clock = pygame.time.Clock()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop when video ends

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
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

class Graphics():
    def __init__(self, screen_setting='fullscreenx'):
        self.screen_setting = screen_setting
        self.x, self.y = None, None
        self.display_surface = None
        self.last_highlight = None
        self.setup()

    def setup(self):
        pygame.init()
        screen = pygame.display.set_mode()
        self.x, self.y = screen.get_size()
        flags = pygame.FULLSCREEN if self.screen_setting.lower() == 'fullscreen' else 0
        self.display_surface = pygame.display.set_mode((self.x, self.y), flags)
        pygame.display.set_caption("OpenCV MarioKart")

    def run(self):
        running = True
        while running:
            self.display_surface.fill((0, 0, 0))

            latest_highlight = load_latest_highlight()

            if latest_highlight and latest_highlight != self.last_highlight:
                self.last_highlight = latest_highlight
                print(f"Playing highlight: {latest_highlight}")
                play_video(self.display_surface, latest_highlight)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False  # Quit if Q or ESC is pressed

            pygame.display.update()
            time.sleep(1)  # Check for new highlights every second

if __name__ == "__main__":
    game = Graphics()
    game.run()


