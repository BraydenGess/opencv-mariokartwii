from curses.textpad import rectangle
import io
import cv2
import sys
import pygame
import time
import numpy as np
from __init__ import *
from graphics.highlights import play_highlights

class Graphics():
    def __init__(self, screen_setting='fulscreen'):
        self.screen_setting = screen_setting
        self.x, self.y = None, None
        self.display_surface = None
        ### Orange, Blue, Red, Green
        self.colors = [(255,165,0), (0,0,255), (255,0,0), (0, 255, 0)]
        self.setup()
        '''
        Special Effects
        '''
        self.red = 0

    def setup(self):
        pygame.init()
        screen = pygame.display.set_mode()
        self.x, self.y = screen.get_size()
        flags = pygame.FULLSCREEN if self.screen_setting.lower() == 'fullscreen' else 0
        self.display_surface = pygame.display.set_mode((self.x, self.y), flags)
        pygame.display.set_caption("OpenCV MarioKart")
        height,width = 1080,1920
        y_margin, x_margin = height//64, width//64
        self.regions_4 = {
            'region0': [0, height//2-y_margin, x_margin, width // 2 - x_margin],
            'region1': [0, height//2-y_margin, width//2, width - x_margin],
            'region2': [height//2, height-y_margin, x_margin, width // 2 - x_margin],
            'region3': [height//2, height-y_margin, width//2, width - x_margin]
        }
        self.regions_2 = {
            'region0': [0, height // 2 - y_margin, x_margin, width - x_margin],
            'region1': [height // 2, height - y_margin, x_margin, width - x_margin],
        }

    '''HELPER FUNCTIONS'''
    def create_text(self,font,font_size,text,color,coordinates,anchor):
        font = pygame.font.SysFont(font,font_size)
        txt = font.render(text,True,color)
        txtRect = txt.get_rect()
        txtRect.center = (int(coordinates[0]),int(coordinates[1]))
        if anchor == 'left':
            txtRect.left = (int(coordinates[0]))
        elif anchor == 'right':
            txtRect.right = (int(coordinates[0]))
        return txt,txtRect
    def write_text(self,texts):
        for element in texts:
            [txt,txtRect] = element
            self.display_surface.blit(txt, txtRect)
    def write_rectangles(self,rectangles):
        for element in rectangles:
            [rect, rgb] = element
            pygame.draw.rect(self.display_surface,rgb,rect,10)
    def string_tocolor(self, string):
        if string == 'white':
            return (255, 255, 255)
        if string.lower() == 'black':
            return (0, 0, 0)
        return (255,255,255)
    '''HELPER FUNCTIONS'''

    def opening(self):
        open_txt, open_box = self.create_text(font = 'chalkduster',font_size=124,text='Beerio Final Cup',
                                              color=(self.red,0,0),coordinates = [self.x//2,self.y//4],anchor = 'c')
        year_txt, year_box = self.create_text(font = 'chalkduster',font_size=72,text='2025',color=(self.red,0,0),
                                              coordinates = [self.x//2,self.y//3],anchor = 'c')
        self.write_text([[open_txt, open_box],[year_txt, year_box]])

    def draw_charts(self, stats, x_buffer, y_buffer, gp):
        max_value = 8 if gp.main_state == 1 else 71
        bottom = self.y - (y_buffer // 2)
        left = x_buffer
        graph_width = self.x // 8
        graph_margin = (self.x - ((graph_width * 7) + (2 * x_buffer))) // 7
        rect_width = graph_width // gp.player_count # hard-coded
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
        x_buffer, y_buffer = self.x // 32, self.y//8

        stats = [[],[],[],[],[],[],[]]
        name_texts = []
        ### Write character Names
        for i in range(gp.player_count):
            character_name = gp.characters[i]
            character_color = self.colors[i]
            anchor = 'left'
            cx = x_buffer
            font_size = 60
            if i%2 == 1:
                anchor = 'right'
                cx = self.x - x_buffer
            cy = ((i//2)*(y_buffer*1.5)) + y_buffer//2
            txt, txt_box = self.create_text(font='Arial', font_size = font_size,text=character_name,
                                            color = character_color,coordinates = [cx, cy], anchor = anchor)
            name_texts.append([txt, txt_box])

            character_stats = gp.characterstats[character_name]
            for j in range(len(character_stats)):
                stats[j].append(int(character_stats[j]))

            if gp.main_state == 2:
                vehicle_name = gp.vehicles[i]
                character_color = self.colors[i]
                display_name = f'[{vehicle_name}]'
                buffer = int(font_size*5/6)
                txt, txt_box = self.create_text(font='Arial', font_size=int(font_size*7/12), text=display_name,
                                                color=character_color,coordinates=[cx, cy+buffer], anchor=anchor)
                name_texts.append([txt, txt_box])

                vehicle_stats = gp.vehiclestats[vehicle_name]
                for k in range(len(vehicle_stats)):
                    stats[k][i] += int(vehicle_stats[k])

        texts, rectangles = self.draw_charts(stats, x_buffer, y_buffer, gp)
        self.write_rectangles(rectangles)
        self.write_text(texts)
        self.write_text(name_texts)

    def course_intro(self, sp, gp):
        if sp.course_queued != None:
            course_name = sp.course_queued
            img_path = f'graphics/assets/CourseImages/{course_name}.png'
            course_image = pygame.image.load(img_path)
            course_image = pygame.transform.scale(course_image,self.display_surface.get_size())
            self.display_surface.blit(course_image, (0, 0))

            [fast_staff, length, AP, CPI, text_color] = gp.course_data[course_name]
            txt, txtRect = self.create_text('impact', 74, course_name, self.string_tocolor(text_color),
                                            (self.x // 45, self.y // 20), 'left')
            txt2, txtRect2 = self.create_text('impact', 24, CPI, self.string_tocolor(text_color),
                                              (self.x // 200, self.y // 40), 'left')
            texts = [[txt, txtRect], [txt2, txtRect2]]

            labels = ['AP','Length','Fast Staff','Best Drinker Odds']
            values = [AP, length, fast_staff,str(round(get_drinkerodds(gp), 2))]
            for i in range(len(labels)):
                label = labels[i]
                value = values[i]
                color = self.string_tocolor(text_color),
                text = f'{label}: {value}' + ('%' if 'Odds' in label else "")
                new_txt,new_txtRect = self.create_text('avenirnextcondensed', 24, text, color,
                                                        (self.x // 45, self.y*(4+i)//40), 'left')
                texts.append([new_txt,new_txtRect])
            self.write_text(texts)

    def song_intro(self, sp):
        if sp.song_queued != None:
            width, height = min(self.x, self.y) // 1.5, min(self.x, self.y) // 1.5
            x_start, y_start = self.x // 2 - (width // 2), self.y // 2 - (height // 2)
            image_file = io.BytesIO(sp.song_img)
            initial_pic = pygame.image.load(image_file)
            desired_size = (width, height)
            pic = pygame.transform.smoothscale(initial_pic, desired_size)
            txt, txtRect = self.create_text('impact', 74, sp.song_queued.song_name, (255, 255, 255),
                                                     (self.x // 2, self.y - self.y // 16),
                                                     'center')
            texts = [[txt, txtRect]]
            self.write_text(texts)
            self.display_surface.blit(pic, (x_start, y_start))


    def live_feed(self, rolling_queue, region, count, gp):
        t1 = time.time()
        if rolling_queue:
            #frame = rolling_queue[-1][0]  # Get the latest frame
            frame = cv2.imread('development/Images/Placement/1+12+11+10_LuigiCircuit_147.png')

            if gp.player_count == 4:
                key = f'region{region}'
                [y0, y1, x0, x1] = self.regions_4[key]
            elif gp.player_count == 2:
                key = f'region{region}'
                [y0, y1, x0, x1] = self.regions_2[key]

            frame = frame[y0:y1, x0:x1]

            # Convert OpenCV BGR frame to RGB and then to Pygame surface
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)
            frame = np.rot90(frame)  # Rotate to match Pygame's coordinate system
            frame_surface = pygame.surfarray.make_surface(frame)

            # Draw the frame on Pygame's display surface
            #frame_surface = pygame.transform.scale(frame_surface, self.display_surface.get_size())
            frame_surface = pygame.transform.scale(frame_surface, (self.x, self.y))

            self.display_surface.blit(frame_surface, (0, 0))

            # Blit the icon to the top-left corner (coordinates (0, 0))
            icon_x, icon_y = self.x//64, self.x//64
            icon_width, icon_height = self.x//8, self.x//8
            border_thickness = 5
            icon = pygame.image.load(f'graphics/assets/CharacterImages/{gp.characters[region]}.png')  # Replace with your icon path
            icon = pygame.transform.scale(icon, (icon_width, icon_height))  # Resize the icon to fit your needs
            pygame.draw.rect(self.display_surface, (255, 255, 255),
                             (icon_x - border_thickness, icon_y - border_thickness,
                              icon_width + 2 * border_thickness, icon_height + 2 * border_thickness))
            self.display_surface.blit(icon, (icon_x,icon_y))
        t2 = time.time()
        print(t2-t1)


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
            if not ret or gp.course_state != 2:
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
                        cap.release()
                        return  # Quit playback
            clock.tick(fps)
        cap.release()


    def racing(self, sp, gp, rolling_queue):
        t2 = time.time()
        if t2-gp.course_start <= 5:
            self.song_intro(sp)
        else:
            time_delta = t2-(gp.course_start+5)
            key = time_delta%50
            feed_time = 20
            if key <= feed_time:
                character = key//(feed_time//gp.player_count)
                self.live_feed(rolling_queue = rolling_queue, region = int(character), count=gp.player_count, gp = gp)
            elif key <= 32:
                video_path = f'graphics/assets/CourseIntros/{sp.course_queued}.mp4'
                self.play_video(self.display_surface,video_path,self.x,self.y, gp)
            elif key <= 50:
                self.course_intro(sp, gp)


    def special_effect(self):
        self.red = min(self.red+0.1, 255)

    def run(self, sp, gp, rolling_queue):
        running = True
        t = time.time()
        while running:

            self.display_surface.fill((0, 0, 0))
            self.special_effect()

            if gp.main_state == 0:
                self.opening()
            elif gp.main_state > 0:
                self.display_stats(gp)
            elif gp.course_state == 1:
                self.course_intro(sp, gp)
            elif gp.course_state == 2:
                self.racing(sp, gp, rolling_queue)
            elif gp.course_state == 3:
                play_highlights(self, self.display_surface, gp, self.x, self.y)
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


def get_drinkerodds(gp):
    history = gp.course_history
    if len(history) >= 3 or len(history) <= 0:
        return 100
    history_num = []
    for course in gp.course_history:
        history_num.append(int(gp.course_data[course][2]))
    last_num = history_num[-1]
    if len(history) == 2:
        worse_left = (32-last_num) - (1 if history_num[0]>=last_num else 0)
        return (worse_left/(32-len(history_num)))*100
    elif len(history) == 1:
        worse_left = (32-last_num)
        return (worse_left/31)*((worse_left-1)/30)*100


if __name__ == "__main__":
    game = Graphics()
