import cv2 as cv
import collections
from collections import deque
import time

def capture_10():
    cap = cv.VideoCapture(0)
    pic = 0
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        count += 1
        file_name = "Rainbow Road"
        cv.imwrite(f'development/Images/RawImages/{file_name}_{pic}.png',frame)
        pic += 1
        if count == 275:
            break

def capture_1():
    cap = cv.VideoCapture(0)
    ret, frame = cap.read()
    file_name = '2characterWario'
    cv.imwrite(f'development/Images/RawImages/{file_name}.png', frame)
    print(file_name)

capture_10()


def spell_fix():
    import os

    # Path to the folder containing the files
    folder_path = 'Images/Courses/Grumble Volcano'

    # Loop through all the files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file matches the pattern 'DSPeachBeach_xx.jpg'
        if filename.startswith('GrumbleVolcanp_') and filename.endswith('.png'):
            # Construct the new filename by replacing 'Beach' with 'Gardens'
            new_filename = filename.replace('GrumbleVolcanp', 'GrumbleVolcano')
            # Get the full path of the current and new file
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_filename)

            # Rename the file
            os.rename(old_file, new_file)
            print(f'Renamed: {filename} -> {new_filename}')


def save_video(frames, filename):
    """
    Saves frames to a video file.
    """
    if not frames:
        return

    fps = 30
    height, width, _ = frames[0].shape
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # MP4 format
    out = cv.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in frames:
        #flipped_frame = cv.flip(frame,0)
        out.write(frame)

    out.release()
    print(f"Saved highlight: {filename}")
