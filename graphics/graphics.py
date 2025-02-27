import pygame

class Graphics():
    def __init__(self, screen_setting = 'fullscreen'):
        self.screen_setting = screen_setting
        self.setup()

    def setup(self):
        screen = pygame.display.set_model()
        self.x, self.y = screen.get_size()
        flags = pygame.FULLSCREEN if self.screen_setting.lower() == 'fullscreen' else 0
        display_surface = pygame.display.set_mode((self.x, self.y), flags)
        pygame.display.set_caption("opencv MarioKart")

