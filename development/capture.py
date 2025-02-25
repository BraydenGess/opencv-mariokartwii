import cv2 as cv

def capture_10():
    cap = cv.VideoCapture(0)
    pic = 0
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        count += 1
        if count % 18 == 0:
            file_name = 'BabyMario+BabyLuigi+BabyPeach+TransRob'
            cv.imwrite(f'development/Images/Characters/{file_name}_{pic}.png',frame)
            pic += 1
        if count == 73:
            break

def capture_1():
    cap = cv.VideoCapture(0)
    ret, frame = cap.read()
    file_name = 'none_15'
    cv.imwrite(f'development/Images/None/None/{file_name}.png', frame)
    print(file_name)

capture_1()


def spell_fix():
    import os

    # Path to the folder containing the files
    folder_path = 'Images/Courses/GrumbleVolcano'

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