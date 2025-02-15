import cv2 as cv

def capture_10():
    cap = cv.VideoCapture(0)
    pic = 0
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        count += 1
        if count % 3 == 0:
            file_name = 'N64BowsersCastle'
            cv.imwrite(f'Images/Courses/{file_name}_{pic}.png',frame)
            pic += 1
        if count == 650:
            break

def capture_1():
    cap = cv.VideoCapture(0)
    ret, frame = cap.read()
    file_name = 'Unlabeled_0'
    cv.imwrite(f'Images/MenuScreen/{file_name}.png', frame)
    print('yes')

capture_1()