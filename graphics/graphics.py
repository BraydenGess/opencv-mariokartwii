from curses.textpad import rectangle
import sys
import pygame
import time
from __init__ import *

class Graphics():
    def __init__(self, screen_setting='fllscreen'):
        self.screen_setting = screen_setting
        self.x, self.y = None, None
        self.display_surface = None
        ### Orange, Blue, Red, Green
        self.colors = [(255,165,0), (0,0,255), (255,0,0), (0, 255, 0)]
        self.setup()

    def setup(self):
        pygame.init()
        screen = pygame.display.set_mode()
        self.x, self.y = screen.get_size()
        flags = pygame.FULLSCREEN if self.screen_setting.lower() == 'fullscreen' else 0
        self.display_surface = pygame.display.set_mode((self.x, self.y), flags)
        pygame.display.set_caption("OpenCV MarioKart")

    '''HELPER FUNCTIONS'''
    def create_text(self,font,font_size,text,color,coordinates,anchor):
        font = pygame.font.SysFont(font,font_size)
        txt = font.render(text,True,color)
        txtRect = txt.get_rect()
        txtRect.center = (coordinates[0],coordinates[1])
        if anchor == 'left':
            txtRect.left = (coordinates[0])
        elif anchor == 'right':
            txtRect.right = (coordinates[0])
        return txt,txtRect
    def write_text(self,texts):
        for element in texts:
            [txt,txtRect] = element
            self.display_surface.blit(txt, txtRect)
    def write_rectangles(self,rectangles):
        for element in rectangles:
            [rect, rgb] = element
            pygame.draw.rect(self.display_surface,rgb,rect,10)
    '''HELPER FUNCTIONS'''

    def opening(self):
        font = pygame.font.Font(None, 74)  # Use default font, size 74
        text = font.render("MarioKart", True, (0, 0, 255))  # White color
        text_rect = text.get_rect(center=(self.x // 2, self.y // 3))
        self.display_surface.blit(text, text_rect)  # Draw text onto the display surface

    def draw_charts(self, stats, x_buffer, y_buffer):
        max_value = max([num for sublist in stats for num in sublist])
        bottom = self.y - (y_buffer // 2)
        left = x_buffer
        graph_width = self.x // 8
        graph_margin = (self.x - ((graph_width * 7) + (2 * x_buffer))) // 7
        rect_width = graph_width // 4 # hard-coded
        rectangles, texts = list(), list()
        labels = ['Speed', 'Weight', 'Acceleration', 'Handle', 'Drift', 'Off-Road', 'Mini-Turbo']
        for i in range(len(stats)):
            xc = left + (graph_width // 2) + (graph_margin * i) + (graph_width * i)
            txt, txtRect = self.create_text('Arial', 32, labels[i], (255, 255, 255),
                                            (xc, bottom + ((self.y - bottom) // 2)), 'center')
            texts.append([txt, txtRect])
            for j in range(len(stats[i])):
                color = self.colors[j]
                x0 = left + (j * rect_width) + (graph_margin * i) + (graph_width * i)
                y0 = bottom
                stats_value = max(0.01, int(stats[i][j]))
                rect_height = max(int((stats_value / max_value) * (self.y // 2 - (y_buffer // 2))), 1)
                rect = pygame.Rect(x0, y0 - rect_height, rect_width, rect_height)
                rectangles.append([rect, color])
        return texts, rectangles

    def display_stats(self, gp):
        x_buffer = self.x // 32
        y_buffer = self.y // 8
        font = pygame.font.Font(None, 36)
        y_offset = 50

        stats = [[],[],[],[],[],[],[]]
        ### Write character Names
        for i in range(len(gp.characters)):
            character_name = gp.characters[i]
            character_color = self.colors[i]
            text = font.render(character_name, True, character_color)  # White text
            text_rect = text.get_rect(topleft=(50, y_offset))  # Position text with some padding
            self.display_surface.blit(text, text_rect)  # Draw text onto the display surface
            y_offset += 40

            character_stats = gp.characterstats[character_name]
            for j in range(len(character_stats)):
                stats[j].append(character_stats[j])

            if gp.main_state == 2:
                vehicle_name = gp.vehicles[i]
                character_color = self.colors[i]
                text = font.render(vehicle_name, True, character_color)  # White text
                text_rect = text.get_rect(topleft=(150, y_offset-40))  # Position text with some padding
                self.display_surface.blit(text, text_rect)

                vehicle_stats = gp.vehiclestats[vehicle_name]
                for k in range(len(vehicle_stats)):
                    stats[k][i] += vehicle_stats[k]

        texts, rectangles = self.draw_charts(stats, x_buffer, y_buffer)
        self.write_rectangles(rectangles)
        self.write_text(texts)


    def course_intro(self, sp):
        if sp .course_queued != None:
            course_name = sp.course_queued
            img_path = f'graphics/assets/CourseImages/{course_name}.png'
            course_image = pygame.image.load(img_path)
            course_image = pygame.transform.scale(course_image,self.display_surface.get_size())
            self.display_surface.blit(course_image, (0, 0))


    def run(self, sp, gp):
        running = True
        while running:
            self.display_surface.fill((0, 0, 0))

            if gp.main_state == 0:
                self.opening()
            elif gp.main_state > 0:
                self.display_stats(gp)
            elif gp.course_state == 0:
                self.course_intro(sp)
                pass
            elif gp.course_state == 1:
                self.display_surface.fill((255, 0, 0))
                # Do course shit
                pass
            elif gp.course_state == 2:
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
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Graphics()
