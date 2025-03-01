import pygame
import time

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


    def run(self, gp):
        running = True
        while running:
            self.display_surface.fill((0, 0, 0))

            if gp.main_state == 0:
                #Beerio Opening
                pass
            elif gp.main_state > 0:
                self.display_surface.fill((0, 0, 0))
                pass
            elif gp.course_state == 1:
                self.display_surface.fill((0, 0, 255))
                #Display course image
                pass
            elif gp.course_state == 2:
                self.display_surface.fill((255, 0, 0))
                # Do course shit
                pass
            elif gp.course_state == 3:
                self.display_surface.fill((0, 255, 0))
                #cours finished
                pass

            pygame.display.update()


            ### Exit handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False  # Quit if Q or ESC is pressed

if __name__ == "__main__":
    game = Graphics()
    game.run()